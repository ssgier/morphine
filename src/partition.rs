use bus::BusReader;
use itertools::Itertools;
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng,
};
use std::hash::{Hash, Hasher};
use std::{collections::hash_map::DefaultHasher, sync::mpsc::Sender as MpscSender};

use crate::{
    batched_ring_buffer::BatchedRingBuffer,
    neuron::Neuron,
    params::{
        InitialSynWeight, InstanceParams, LayerConnectionParams, NeuronParams, ProjectionParams,
    },
    plasticity_modulation::PlasticityModulator,
    short_term_plasticity::{self, ShortTermPlasticity},
    spike_coincidence::SpikeCoincidence,
    state_snapshot::{NeuronState, SynapseState},
    synapse::Synapse,
    types::HashMap,
    util,
};

pub struct PartitionGroupResult {
    pub spiking_nids: Vec<usize>,
    pub partition_state_snapshots: Option<Vec<PartitionStateSnapshot>>,
    pub synaptic_transmission_count: usize,
}

pub struct PartitionStateSnapshot {
    pub nid_start: usize,
    pub neuron_states: Vec<NeuronState>,
    pub synapse_states: Vec<SynapseState>,
}

pub struct Partition {
    nid_start: usize,
    nid_to_projection: HashMap<usize, Projection>,
    neuron_params: NeuronParams,
    neurons: Vec<Neuron>,
    neuron_indexes_to_check: Vec<usize>,
    transmission_buffer: BatchedRingBuffer<TransmissionEvent>,
    plasticity_modulator: Option<PlasticityModulator>,
}

struct Projection {
    synapses: Vec<Synapse>,
    stp: Box<dyn ShortTermPlasticity>,
    prj_params: ProjectionParams,
    last_pre_syn_spike_t: Option<usize>,
    next_to_last_pre_syn_spike_t: Option<usize>,
}

#[derive(Debug, Clone)]
struct TransmissionEvent {
    neuron_idx: usize,
    pre_syn_nid: usize,
    syn_idx: usize,
    psp: f32,
}

pub fn create_partitions(
    num_threads: usize,
    thread_id: usize,
    params: &InstanceParams,
) -> Vec<Partition> {
    let mut partitions = Vec::new();

    let mut to_layer_id_to_conn_params = HashMap::default();

    for conn in &params.layer_connections {
        to_layer_id_to_conn_params
            .entry(conn.to_layer_id)
            .or_insert_with(Vec::new)
            .push(conn);
    }

    let mut layer_nid_starts = Vec::new();

    let mut next_layer_nid_start = 0;
    for layer_params in &params.layers {
        layer_nid_starts.push(next_layer_nid_start);
        next_layer_nid_start += layer_params.num_neurons;
    }

    let seed = params.technical_params.seed_override.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);

    for (layer_id, layer_params) in params.layers.iter().enumerate() {
        let mut nid_to_projection = HashMap::default();
        let mut neurons = Vec::new();

        let partition_range =
            util::get_partition_range(num_threads, thread_id, layer_params.num_neurons);

        let nid_start = layer_nid_starts[layer_id] + partition_range.start;

        for _ in 0..partition_range.len() {
            neurons.push(Neuron::new());
        }

        let plasticity_modulator = layer_params
            .plasticity_modulation_params
            .as_ref()
            .map(|params| PlasticityModulator::new(params.clone()));

        if let Some(connection_params_elements) = to_layer_id_to_conn_params.get(&layer_id) {
            for connection_params in connection_params_elements.iter() {
                let from_num_neurons = params.layers[connection_params.from_layer_id].num_neurons;
                let to_num_neurons = params.layers[connection_params.to_layer_id].num_neurons;

                let from_positions = get_positions_1d(from_num_neurons);
                let to_positions = get_positions_1d(to_num_neurons);

                let to_delta = 1.0 / ((to_num_neurons - 1) as f64);
                let to_indexes = Vec::from_iter(0..to_num_neurons);

                for from_idx in 0..from_num_neurons {
                    let from_nid = layer_nid_starts[connection_params.from_layer_id] + from_idx;
                    let from_pos = from_positions[from_idx];

                    let to_pos_lower_bound = from_pos - 0.5 * connection_params.connect_width;
                    let to_pos_upper_bound = from_pos + 0.5 * connection_params.connect_width;

                    let to_idx_lower_bound =
                        (((to_pos_lower_bound - f64::EPSILON) / to_delta).ceil() as usize).max(0);
                    let to_idx_upper_bound = (((to_pos_upper_bound + f64::EPSILON) / to_delta)
                        .floor() as usize)
                        .min(to_num_neurons - 1);

                    if to_idx_upper_bound < to_idx_lower_bound {
                        continue;
                    }

                    let target_slice = &to_indexes[to_idx_lower_bound..=to_idx_upper_bound];

                    let num_targets = (target_slice.len() as f64
                        * connection_params.connect_density)
                        .round() as usize;

                    let mut synapses = Vec::new();

                    // seed generators in such a way that the result is independent of the number of threads
                    for to_idx in target_slice.choose_multiple(&mut rng, num_targets) {
                        if partition_range.contains(to_idx) {
                            let to_pos = to_positions[*to_idx];

                            let position_distance = (to_pos - from_pos).abs();

                            let neuron_idx = *to_idx - partition_range.start;

                            let post_syn_nid = neuron_idx + nid_start;
                            let pre_syn_nid = from_nid;

                            let mut rng = StdRng::seed_from_u64(calculate_hash(&(
                                seed,
                                pre_syn_nid,
                                post_syn_nid,
                            )));

                            let conduction_delay = compute_conduction_delay(
                                &connection_params,
                                position_distance,
                                &mut rng,
                            );

                            let init_weight = compute_initial_weight(
                                &connection_params.initial_syn_weight,
                                &mut rng,
                            );

                            if pre_syn_nid != post_syn_nid
                                || connection_params.allow_self_innervation
                            {
                                let synapse =
                                    Synapse::new(neuron_idx, conduction_delay as u8, init_weight);

                                synapses.push(synapse);
                            }
                        }
                    }

                    if !synapses.is_empty() {
                        synapses.sort_by_key(|synapse| synapse.neuron_idx);

                        let projection = Projection {
                            synapses,
                            stp: short_term_plasticity::create(
                                &connection_params.projection_params.stp_params,
                            ),
                            prj_params: connection_params.projection_params.clone(),
                            last_pre_syn_spike_t: None,
                            next_to_last_pre_syn_spike_t: None,
                        };

                        nid_to_projection.insert(from_nid, projection);
                    }
                }
            }
        }

        let max_conduction_delay = nid_to_projection
            .values()
            .map(|projection| projection.synapses.iter())
            .flatten()
            .map(|synapse| synapse.conduction_delay)
            .max()
            .unwrap_or(0);

        let batched_ring_buffer_size = max_conduction_delay as usize + 1;

        let transmission_buffer = BatchedRingBuffer::new(batched_ring_buffer_size);

        let partition = Partition {
            nid_start,
            nid_to_projection,
            neuron_params: layer_params.neuron_params.clone(),
            neurons,
            neuron_indexes_to_check: Vec::new(),
            transmission_buffer,
            plasticity_modulator,
        };

        partitions.push(partition);
    }

    partitions
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn compute_initial_weight(init_syn_weight: &InitialSynWeight, rng: &mut StdRng) -> f32 {
    match init_syn_weight {
        InitialSynWeight::Randomized(max_init_weight) => {
            Uniform::new_inclusive(0.0, max_init_weight).sample(rng)
        }
        InitialSynWeight::Constant(init_weight) => *init_weight,
    }
}

fn compute_conduction_delay(
    connection_params: &LayerConnectionParams,
    position_distance: f64,
    rng: &mut StdRng,
) -> u8 {
    let mut result = 1;

    let random_part =
        Uniform::from(0..=(connection_params.conduction_delay_max_random_part)).sample(rng);

    result += random_part;

    result += (position_distance
        * connection_params.conduction_delay_position_distance_scale_factor)
        .round() as usize
        + connection_params.conduction_delay_add_on;

    result as u8
}

fn get_positions_1d(num_neurons: usize) -> Vec<f64> {
    if num_neurons == 1 {
        vec![0.5]
    } else {
        let delta = 1.0 / ((num_neurons - 1) as f64);

        let mut positions = Vec::new();

        for i in 0..num_neurons {
            let position = delta * (i as f64);
            positions.push(position);
        }

        positions
    }
}

impl Partition {
    pub fn run(
        partitions: &mut [Partition],
        mut rx: BusReader<TickContext>,
        partition_result_tx: MpscSender<PartitionGroupResult>,
    ) {
        while let Ok(ctx) = rx.recv() {
            let mut spiking_nids = Vec::new();
            let mut synaptic_transmission_count = 0;

            for partition in partitions.iter_mut() {
                partition.process_tick(&ctx, &mut spiking_nids, &mut synaptic_transmission_count);
            }

            let partition_state_snapshots = if ctx.extract_state_snapshot {
                let mut snapshots = Vec::new();
                for partition in &mut *partitions {
                    snapshots.push(partition.extract_state_snapshot(&ctx));
                }
                Some(snapshots)
            } else {
                None
            };

            partition_result_tx
                .send(PartitionGroupResult {
                    spiking_nids,
                    partition_state_snapshots,
                    synaptic_transmission_count,
                })
                .ok();
        }
    }

    fn extract_state_snapshot(&self, ctx: &TickContext) -> PartitionStateSnapshot {
        let neuron_states = self
            .neurons
            .iter()
            .map(|neuron| NeuronState {
                voltage: neuron.get_voltage(ctx.t, &self.neuron_params),
            })
            .collect();

        let synapse_states = self
            .nid_to_projection
            .iter()
            .sorted_by_key(|entry| entry.0) // sort by nid
            .map(|(pre_syn_nid, projection)| {
                projection.synapses.iter().map(|synapse| SynapseState {
                    pre_syn_nid: *pre_syn_nid,
                    post_syn_nid: self.nid_start + synapse.neuron_idx,
                    conduction_delay: synapse.conduction_delay,
                    weight: synapse.weight,
                })
            })
            .flatten()
            .collect::<Vec<_>>();

        PartitionStateSnapshot {
            nid_start: self.nid_start,
            neuron_states,
            synapse_states,
        }
    }

    fn process_tick(
        &mut self,
        ctx: &TickContext,
        spiking_nids: &mut Vec<usize>,
        synaptic_transmission_count: &mut usize,
    ) {
        self.process_modulated_plasticity(ctx.t, ctx.dopamine_amount);
        if ctx.t > 0 {
            self.process_spikes(&ctx);
        }
        self.process_transmission_events(ctx.t, synaptic_transmission_count);
        spiking_nids.extend(self.check_for_spikes(ctx));
    }

    #[cfg(test)]
    fn get_num_neurons(&self) -> usize {
        self.neurons.len()
    }

    fn process_modulated_plasticity(&mut self, t: usize, dopamine_amount: f32) {
        if let Some(plasticity_modulator) = &mut self.plasticity_modulator {
            plasticity_modulator.process_dopamine(dopamine_amount);
            if let Some(plasticity_events) = plasticity_modulator.tick(t) {
                for event in plasticity_events.iter() {
                    let projection = self.nid_to_projection.get_mut(&event.pre_syn_nid).unwrap();

                    let synapse = &mut projection.synapses[event.synapse_idx];

                    synapse.process_weight_change(
                        event.weight_change,
                        &projection.prj_params.synapse_params,
                    );
                }
            }
        }
    }

    fn process_spikes(&mut self, ctx: &TickContext) {
        let spike_t = ctx.t - 1;
        for nid in &ctx.spiked_nids {
            self.process_spike(spike_t, *nid);
        }
    }

    fn check_for_spikes(&mut self, ctx: &TickContext) -> Vec<usize> {
        let mut spiking_nids = Vec::new();

        for spike_trigger_nid in &ctx.spike_trigger_nids {
            if *spike_trigger_nid >= self.nid_start {
                let neuron_idx = spike_trigger_nid - self.nid_start;

                if neuron_idx < self.neurons.len() {
                    let spike = self.neurons[neuron_idx].spike(ctx.t, &self.neuron_params);
                    for spike_coincidence in spike.0 {
                        Self::process_spike_coincidence(
                            &mut self.nid_to_projection,
                            &mut self.plasticity_modulator,
                            ctx.t,
                            spike_coincidence,
                        );
                    }

                    spiking_nids.push(*spike_trigger_nid);
                }
            }
        }

        for neuron_idx in self.neuron_indexes_to_check.drain(..) {
            if let Some(spike) = self.neurons[neuron_idx].check_spike(ctx.t, &self.neuron_params) {
                for spike_coincidence in spike.0 {
                    Self::process_spike_coincidence(
                        &mut self.nid_to_projection,
                        &mut self.plasticity_modulator,
                        ctx.t,
                        spike_coincidence,
                    );
                }

                spiking_nids.push(neuron_idx + self.nid_start);
            }
        }

        spiking_nids
    }

    fn process_spike_coincidence(
        nid_to_projection: &mut HashMap<usize, Projection>,
        plasticity_modulator: &mut Option<PlasticityModulator>,
        t: usize,
        spike_coincidence: SpikeCoincidence,
    ) {
        let projection = nid_to_projection
            .get_mut(&spike_coincidence.pre_syn_nid)
            .unwrap();

        if let Some(short_term_stdp_params) = &projection.prj_params.short_term_stdp_params {
            let stdp_value = util::compute_stdp(
                spike_coincidence.t_pre_minus_post,
                &short_term_stdp_params.stdp_params,
            );

            projection.synapses[spike_coincidence.syn_idx].process_short_term_stdp(
                t,
                stdp_value,
                short_term_stdp_params.tau,
            );
        }

        if let Some(long_term_stdp_params) = &projection.prj_params.long_term_stdp_params {
            let stdp_value =
                util::compute_stdp(spike_coincidence.t_pre_minus_post, long_term_stdp_params);

            if let Some(plasticity_modulator) = plasticity_modulator {
                plasticity_modulator.process_stdp_value(
                    t,
                    spike_coincidence.pre_syn_nid,
                    spike_coincidence.syn_idx,
                    stdp_value,
                );
            } else {
                projection.synapses[spike_coincidence.syn_idx]
                    .process_weight_change(stdp_value, &projection.prj_params.synapse_params)
            }
        }
    }

    fn process_transmission_events(&mut self, t: usize, synaptic_transmission_count: &mut usize) {
        *synaptic_transmission_count += self.transmission_buffer.next_batch_size();

        for transm_event in self.transmission_buffer.drain_and_advance() {
            let psp_result = self.neurons[transm_event.neuron_idx].apply_psp(
                t,
                transm_event.psp,
                transm_event.pre_syn_nid,
                transm_event.syn_idx,
                &self.neuron_params,
            );

            if psp_result.might_spike {
                self.neuron_indexes_to_check.push(transm_event.neuron_idx);
            }

            if let Some(spike_coincidence) = psp_result.spike_coincidence {
                let next_to_last_pre_syn_spike_t =
                    self.nid_to_projection[&transm_event.pre_syn_nid].next_to_last_pre_syn_spike_t;

                let last_post_syn_spike_t =
                    self.neurons[transm_event.neuron_idx].get_last_spike_t();

                let is_eligible =
                    if let (Some(next_to_last_pre_syn_spike_t), Some(last_post_syn_spike_t)) =
                        (next_to_last_pre_syn_spike_t, last_post_syn_spike_t)
                    {
                        let next_to_last_transm_t = next_to_last_pre_syn_spike_t
                            + self.nid_to_projection[&transm_event.pre_syn_nid].synapses
                                [transm_event.syn_idx]
                                .conduction_delay as usize;

                        next_to_last_transm_t <= last_post_syn_spike_t
                    } else {
                        true
                    };

                if is_eligible {
                    Self::process_spike_coincidence(
                        &mut self.nid_to_projection,
                        &mut self.plasticity_modulator,
                        t,
                        spike_coincidence,
                    );
                }
            }
        }
    }

    fn process_spike(&mut self, spike_t: usize, spiking_nid: usize) {
        if let Some(projection) = self.nid_to_projection.get_mut(&spiking_nid) {
            projection.next_to_last_pre_syn_spike_t = projection.last_pre_syn_spike_t;
            projection.last_pre_syn_spike_t = Some(spike_t);

            let stp_value = projection.stp.on_pre_syn_spike_get_value(spike_t);

            let short_term_stdp_tau = projection
                .prj_params
                .short_term_stdp_params
                .as_ref()
                .map(|short_term_stdp_params| short_term_stdp_params.tau)
                .unwrap_or(1.0); // if none, default won't have any effect

            for (syn_idx, synapse) in projection.synapses.iter_mut().enumerate() {
                let psp = synapse.process_pre_syn_spike_get_psp(
                    spike_t,
                    stp_value,
                    &projection.prj_params.synapse_params,
                    short_term_stdp_tau,
                );

                let transmission_event = TransmissionEvent {
                    syn_idx,
                    pre_syn_nid: spiking_nid,
                    neuron_idx: synapse.neuron_idx,
                    psp,
                };

                self.transmission_buffer
                    .push_at_offset(synapse.conduction_delay as usize - 1, transmission_event);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TickContext {
    pub t: usize,
    pub spike_trigger_nids: Vec<usize>,
    pub spiked_nids: Vec<usize>,
    pub dopamine_amount: f32,
    pub extract_state_snapshot: bool,
}

#[cfg(test)]
mod tests {
    use crate::{
        params::{LayerParams, TechnicalParams},
        types::HashSet,
    };

    use super::*;
    use crate::util::test_util::assert_approx_eq_slice;
    use itertools::assert_equal;
    use rand::SeedableRng;

    #[test]
    fn conduction_delay() {
        let mut conn_params = LayerConnectionParams {
            from_layer_id: 5,
            to_layer_id: 3,
            projection_params: ProjectionParams::default(),
            connect_density: 1.0,
            connect_width: 1.0,
            initial_syn_weight: InitialSynWeight::Constant(0.2),
            conduction_delay_max_random_part: 0,
            conduction_delay_position_distance_scale_factor: 5.0,
            conduction_delay_add_on: 4,
            allow_self_innervation: true,
        };

        let mut rng = StdRng::seed_from_u64(0);

        let conduction_delay = compute_conduction_delay(&conn_params, 0.5, &mut rng);
        assert_eq!(conduction_delay, 8);

        std::mem::swap(&mut conn_params.from_layer_id, &mut conn_params.to_layer_id);

        let conduction_delay = compute_conduction_delay(&conn_params, 0.5, &mut rng);
        assert_eq!(conduction_delay, 8);

        conn_params.conduction_delay_max_random_part = 2;

        let mut distinct_conduction_delays = HashSet::default();

        for _ in 0..100 {
            let conduction_delay = compute_conduction_delay(&conn_params, 0.5, &mut rng);
            distinct_conduction_delays.insert(conduction_delay);
        }

        assert_eq!(distinct_conduction_delays, HashSet::from_iter([8, 9, 10]));
    }

    #[test]
    fn multithreading_partitions() {
        let mut layers = Vec::new();

        let mut layer_params = LayerParams::default();
        layer_params.num_neurons = 100;

        layers.push(layer_params.clone());
        layer_params.num_neurons = 50;

        layers.push(layer_params);

        let connection_params = LayerConnectionParams {
            from_layer_id: 0,
            to_layer_id: 1,
            projection_params: ProjectionParams::default(),
            connect_density: 1.0,
            connect_width: 2.0,
            initial_syn_weight: InitialSynWeight::Constant(0.5),
            conduction_delay_max_random_part: 0,
            conduction_delay_position_distance_scale_factor: 2.0,
            conduction_delay_add_on: 0,
            allow_self_innervation: true,
        };

        let instance_params = InstanceParams {
            layers,
            layer_connections: vec![connection_params],
            technical_params: TechnicalParams::default(),
        };

        let partitions = create_partitions(3, 2, &instance_params);

        assert_eq!(partitions.len(), 2);

        assert_eq!(partitions[0].nid_start, 67);
        assert_eq!(partitions[1].nid_start, 134);

        assert_eq!(partitions[0].get_num_neurons(), 33);
        assert_eq!(partitions[1].get_num_neurons(), 16);

        assert!(partitions[0].nid_to_projection.is_empty());
        assert_eq!(partitions[1].nid_to_projection.len(), 100);

        assert_eq!(partitions[1].nid_to_projection[&0].synapses.len(), 16);
        assert_eq!(
            partitions[1].nid_to_projection[&0].synapses[0].neuron_idx,
            0
        );
        assert_eq!(
            partitions[1].nid_to_projection[&0].synapses[0].conduction_delay,
            2
        );

        assert_eq!(
            partitions[1].nid_to_projection[&0].synapses[15].neuron_idx,
            15
        );
        assert_eq!(
            partitions[1].nid_to_projection[&0].synapses[15].conduction_delay,
            3
        );
    }

    #[test]
    fn randomized_weights() {
        let layer = LayerParams {
            num_neurons: 100,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
        };

        let conn_params = LayerConnectionParams {
            from_layer_id: 0,
            to_layer_id: 0,
            projection_params: ProjectionParams::default(),
            connect_density: 1.0,
            connect_width: 2.0,
            initial_syn_weight: InitialSynWeight::Randomized(0.2),
            conduction_delay_max_random_part: 0,
            conduction_delay_position_distance_scale_factor: 5.0,
            conduction_delay_add_on: 0,
            allow_self_innervation: true,
        };

        let mut params = InstanceParams::default();
        params.layers.push(layer);
        params.layer_connections.push(conn_params);

        let partitions = create_partitions(1, 0, &params);

        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].nid_to_projection.len(), 100);

        for (_, projection) in &partitions[0].nid_to_projection {
            assert_eq!(projection.synapses.len(), 100);

            assert!(projection
                .synapses
                .iter()
                .any(|synapse| synapse.weight > 0.1));

            assert!(projection
                .synapses
                .iter()
                .any(|synapse| synapse.weight < 0.1));
        }
    }

    #[test]
    fn positions_1d() {
        assert_approx_eq_slice(&get_positions_1d(1), &vec![0.5]);
        assert_approx_eq_slice(&get_positions_1d(2), &vec![0.0, 1.0]);
        assert_approx_eq_slice(&get_positions_1d(3), &vec![0.0, 0.5, 1.0]);
        assert_approx_eq_slice(&get_positions_1d(4), &vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn partition_creation() {
        let mut params = InstanceParams::default();

        let layer_0 = LayerParams {
            num_neurons: 5,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
        };

        let layer_1 = LayerParams {
            num_neurons: 100,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
        };

        let layer_2 = LayerParams {
            num_neurons: 10,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
        };

        params.layers.push(layer_0);
        params.layers.push(layer_1);
        params.layers.push(layer_2);

        let connection_params = LayerConnectionParams {
            from_layer_id: 1,
            to_layer_id: 2,
            projection_params: ProjectionParams::default(),
            connect_density: 1.0,
            connect_width: 2.0,
            initial_syn_weight: InitialSynWeight::Constant(0.2),
            conduction_delay_max_random_part: 0,
            conduction_delay_position_distance_scale_factor: 10.0,
            conduction_delay_add_on: 5,
            allow_self_innervation: true,
        };

        params.layer_connections.push(connection_params);

        // full connection
        let partitions = create_partitions(1, 0, &params);

        assert_eq!(partitions.len(), 3);

        assert_eq!(partitions[0].nid_start, 0);
        assert_eq!(partitions[1].nid_start, 5);
        assert_eq!(partitions[2].nid_start, 105);

        assert_eq!(partitions[0].get_num_neurons(), 5);
        assert_eq!(partitions[1].get_num_neurons(), 100);
        assert_eq!(partitions[2].get_num_neurons(), 10);

        assert!(partitions[0].nid_to_projection.is_empty());
        assert!(partitions[1].nid_to_projection.is_empty());
        assert_eq!(partitions[2].nid_to_projection.len(), 100);

        for nid in 5..105 {
            assert!(partitions[2].nid_to_projection.contains_key(&nid));
            assert_equal(
                partitions[2].nid_to_projection[&nid]
                    .synapses
                    .iter()
                    .map(|synapse| synapse.neuron_idx),
                0..10,
            );
        }

        let synapses_for_nid_65 = &partitions[2].nid_to_projection[&(65)].synapses;

        // position is ~= 0.6060606
        // min delay: 1
        // cross layer delay: 5.0 * 1 = 5.0

        // position diff delay: 0.60606 * 10 = 6.0606
        // total = 12
        assert_eq!(synapses_for_nid_65[0].conduction_delay, 12);

        // position diff delay: 0.05051 * 10 = 0.5051
        // total = 7
        assert_eq!(synapses_for_nid_65[5].conduction_delay, 7);

        // position diff delay: 0.393939 * 10 = 3.93939
        // total = 10
        assert_eq!(synapses_for_nid_65[9].conduction_delay, 10);

        // narrow connection
        params.layer_connections[0].connect_width = 0.4;
        let partitions = create_partitions(1, 0, &params);

        assert_eq!(partitions[2].nid_to_projection.len(), 100);

        for nid in 5..105 {
            assert!(partitions[2].nid_to_projection.contains_key(&nid));
        }

        assert_equal(
            partitions[2].nid_to_projection[&5]
                .synapses
                .iter()
                .map(|synapse| synapse.neuron_idx),
            0..2,
        );

        assert_equal(
            partitions[2].nid_to_projection[&104]
                .synapses
                .iter()
                .map(|synapse| synapse.neuron_idx),
            8..10,
        );

        assert_equal(
            partitions[2].nid_to_projection[&(5 + 60)]
                .synapses
                .iter()
                .map(|synapse| synapse.neuron_idx),
            4..8,
        );

        // sparse connection
        params.layer_connections[0].connect_density = 0.25;
        let partitions = create_partitions(1, 0, &params);

        assert_eq!(partitions[2].nid_to_projection.len(), 100);

        for nid in 5..105 {
            assert!(partitions[2].nid_to_projection.contains_key(&nid));
            assert_eq!(partitions[2].nid_to_projection[&nid].synapses.len(), 1);
        }
    }
}
