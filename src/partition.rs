use crate::api::SCMode;
use crate::distance::DistanceCalculator;
use crate::partition::Request::ExtractStateSnapshot;
use crate::partition::Request::FlushSCHashes;
use crate::partition::Request::ResetEphemeralState;
use crate::partition::Request::SetSCMode;
use crate::partition::Request::Tick;
use bus::BusReader;
use itertools::Itertools;
use rand::distributions::Bernoulli;
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::{collections::hash_map::DefaultHasher, sync::mpsc::Sender as MpscSender};

use crate::util::SynapseCoordinate;
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

#[derive(Debug, Clone)]
pub enum Request {
    Tick(TickContext),
    ExtractStateSnapshot { t: usize },
    ResetEphemeralState { t: usize },
    SetSCMode(SCMode),
    FlushSCHashes,
}

#[derive(Debug, Clone)]
pub struct TickContext {
    pub t: usize,
    pub spike_trigger_nids: Vec<usize>,
    pub spiked_nids: Vec<usize>,
    pub dopamine_amount: f32,
}

pub struct PartitionGroupResult {
    pub spiking_nids: Vec<usize>,
    pub synaptic_transmission_count: usize,
}

pub struct PartitionStateSnapshot {
    pub nid_start: usize,
    pub neuron_states: Vec<NeuronState>,
    pub synapse_states: Vec<SynapseState>,
}

pub struct Partition {
    nid_start: usize,
    nid_to_projections: HashMap<usize, Vec<Projection>>,
    neuron_params: NeuronParams,
    neurons: Vec<Neuron>,
    para_neurons: Option<Vec<Neuron>>,
    neuron_indexes_to_check: Vec<usize>,
    transmission_buffer: BatchedRingBuffer<TransmissionEvent>,
    plasticity_modulator: Option<PlasticityModulator>,
    sc_hash_buffer: HashSet<u64>,
    sc_mode: SCMode,
}

struct Projection {
    synapses: Vec<Synapse>,
    stp: Box<dyn ShortTermPlasticity>,
    prj_params: ProjectionParams,
    last_pre_syn_spike_t: Option<usize>,
    next_to_last_pre_syn_spike_t: Option<usize>,
    projection_id: usize,
}

#[derive(Debug, Clone)]
struct TransmissionEvent {
    neuron_idx: usize,
    syn_coord: SynapseCoordinate,
    psp: f32,
    para_psp: f32,
}

struct ProcessSpikeCoincidenceResult {
    syn_weight: f32,
    max_weight: f32,
    weight_scale_factor: f32,
}

impl ProcessSpikeCoincidenceResult {
    fn is_sc_contrib(&self, threshold: f32) -> bool {
        self.weight_scale_factor > 0.0 && self.syn_weight / self.max_weight >= threshold
    }
}

pub fn create_partitions(
    num_threads: usize,
    thread_id: usize,
    params: &InstanceParams,
) -> Vec<Partition> {
    let mut partitions = Vec::new();

    let mut to_layer_id_to_prj_id_and_conn_params = HashMap::default();

    for conn in params.layer_connections.iter().enumerate() {
        to_layer_id_to_prj_id_and_conn_params
            .entry(conn.1.to_layer_id)
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

    for (layer_id, layer_params) in params.layers.iter().enumerate() {
        let mut nid_to_projections = HashMap::default();
        let mut neurons = Vec::new();

        let partition_range =
            util::get_partition_range(num_threads, thread_id, layer_params.num_neurons);

        let nid_start = layer_nid_starts[layer_id] + partition_range.start;

        for _ in 0..partition_range.len() {
            neurons.push(Neuron::new());
        }

        let para_neurons = if layer_params.use_para_spikes {
            let mut para_neurons = Vec::new();
            for _ in 0..partition_range.len() {
                para_neurons.push(Neuron::new());
            }
            Some(para_neurons)
        } else {
            None
        };

        let plasticity_modulator = layer_params
            .plasticity_modulation_params
            .as_ref()
            .map(|params| PlasticityModulator::new(params.clone()));

        if let Some(connection_params_elements) =
            to_layer_id_to_prj_id_and_conn_params.get(&layer_id)
        {
            for (projection_id, connection_params) in connection_params_elements.iter() {
                let to_num_neurons = params.layers[connection_params.to_layer_id].num_neurons;
                if to_num_neurons == 0 {
                    continue;
                }

                let from_num_neurons = params.layers[connection_params.from_layer_id].num_neurons;

                let distance_calc = DistanceCalculator::new(
                    params.position_dim,
                    params.hyper_sphere,
                    from_num_neurons,
                    to_num_neurons,
                );

                for from_idx in 0..from_num_neurons {
                    let from_nid = layer_nid_starts[connection_params.from_layer_id] + from_idx;

                    let mut synapses = Vec::new();

                    for to_idx in 0..to_num_neurons {
                        if partition_range.contains(&to_idx) {
                            let position_distance =
                                distance_calc.calculate_distance(from_idx, to_idx);

                            let connection_probability = util::compute_connection_probabiltity(
                                position_distance,
                                connection_params.connect_width,
                                connection_params.smooth_connect_probability,
                            );

                            let neuron_idx = to_idx - partition_range.start;

                            let post_syn_nid = neuron_idx + nid_start;
                            let pre_syn_nid = from_nid;

                            // seed generators in such a way that the result is independent of the number of threads
                            let mut rng = StdRng::seed_from_u64(calculate_hash(&(
                                seed,
                                pre_syn_nid,
                                post_syn_nid,
                                projection_id,
                            )));

                            let is_connection = Bernoulli::new(connection_probability)
                                .unwrap()
                                .sample(&mut rng);

                            if is_connection {
                                let conduction_delay = compute_conduction_delay(
                                    connection_params,
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
                                        Synapse::new(neuron_idx, conduction_delay, init_weight);

                                    synapses.push(synapse);
                                }
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
                            projection_id: *projection_id,
                        };

                        nid_to_projections
                            .entry(from_nid)
                            .or_insert_with(Vec::default)
                            .push(projection);
                    }
                }
            }
        }

        let max_conduction_delay = nid_to_projections
            .values()
            .flat_map(|projections| {
                projections
                    .iter()
                    .flat_map(|projection| projection.synapses.iter())
            })
            .map(|synapse| synapse.conduction_delay)
            .max()
            .unwrap_or(0);

        let batched_ring_buffer_size = max_conduction_delay as usize + 1;

        let transmission_buffer = BatchedRingBuffer::new(batched_ring_buffer_size);

        let partition = Partition {
            nid_start,
            nid_to_projections,
            neuron_params: layer_params.neuron_params.clone(),
            neurons,
            para_neurons,
            neuron_indexes_to_check: Vec::new(),
            transmission_buffer,
            plasticity_modulator,
            sc_hash_buffer: HashSet::new(),
            sc_mode: SCMode::Off,
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

impl Partition {
    pub fn run(
        partitions: &mut [Partition],
        mut rx: BusReader<Request>,
        partition_result_tx: MpscSender<PartitionGroupResult>,
        partition_snapshots_tx: MpscSender<Vec<PartitionStateSnapshot>>,
        partition_ack_tx: MpscSender<()>,
        partition_sc_hashes_tx: MpscSender<HashSet<u64>>,
    ) {
        while let Ok(request) = rx.recv() {
            match request {
                Tick(ctx) => {
                    let mut spiking_nids = Vec::new();
                    let mut synaptic_transmission_count = 0;

                    for partition in &mut *partitions {
                        partition.process_tick(
                            &ctx,
                            &mut spiking_nids,
                            &mut synaptic_transmission_count,
                        );
                    }

                    partition_result_tx
                        .send(PartitionGroupResult {
                            spiking_nids,
                            synaptic_transmission_count,
                        })
                        .unwrap();
                }
                ExtractStateSnapshot { t } => {
                    let mut snapshots = Vec::new();
                    for partition in &mut *partitions {
                        snapshots.push(partition.extract_state_snapshot(t));
                    }
                    partition_snapshots_tx.send(snapshots).unwrap();
                }
                ResetEphemeralState { t } => {
                    for partition in &mut *partitions {
                        partition.reset_ephemeral_state(t);
                    }
                    partition_ack_tx.send(()).unwrap();
                }
                SetSCMode(sc_mode) => {
                    for partition in &mut *partitions {
                        partition.sc_mode = sc_mode;
                    }
                }
                FlushSCHashes => {
                    let mut sc_hashes = HashSet::new();
                    for partition in partitions.iter_mut() {
                        sc_hashes.extend(&mut partition.sc_hash_buffer.drain());
                    }
                    partition_sc_hashes_tx.send(sc_hashes).unwrap();
                }
            }
        }
    }

    fn reset_ephemeral_state(&mut self, t: usize) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset_ephemeral_state(t);
        }

        if let Some(para_neurons) = &mut self.para_neurons {
            for para_neuron in para_neurons {
                para_neuron.reset_ephemeral_state(t);
            }
        }

        for projection in self
            .nid_to_projections
            .values_mut()
            .flat_map(|prj| prj.iter_mut())
        {
            projection.stp.reset_ephemeral_state();
            projection.last_pre_syn_spike_t = None;
            projection.next_to_last_pre_syn_spike_t = None;

            for synapse in projection.synapses.iter_mut() {
                synapse.reset_ephemeral_state(t);
            }
        }

        if let Some(modulator) = &mut self.plasticity_modulator {
            modulator.reset_ephemeral_state();
        }

        self.transmission_buffer.clear();
        self.sc_hash_buffer.clear();
    }

    fn extract_state_snapshot(&self, t: usize) -> PartitionStateSnapshot {
        let neuron_states = self
            .neurons
            .iter()
            .map(|neuron| NeuronState {
                voltage: neuron.get_voltage(t, &self.neuron_params),
                threshold: neuron.get_threshold(t, &self.neuron_params),
                is_refractory: neuron.is_refractory(t),
            })
            .collect();

        let synapse_states = self
            .nid_to_projections
            .iter()
            .sorted_by_key(|entry| entry.0) // sort by nid
            .flat_map(|(pre_syn_nid, projections)| {
                projections.iter().flat_map(|projection| {
                    projection.synapses.iter().map(|synapse| {
                        let short_term_stdp_offset = if let Some(short_term_stdp_params) =
                            &projection.prj_params.short_term_stdp_params
                        {
                            synapse.get_short_term_stdp_offset(t, short_term_stdp_params.tau)
                        } else {
                            0.0
                        };

                        SynapseState {
                            pre_syn_nid: *pre_syn_nid,
                            post_syn_nid: self.nid_start + synapse.neuron_idx,
                            conduction_delay: synapse.conduction_delay,
                            weight: synapse.weight,
                            short_term_stdp_offset,
                            projection_id: projection.projection_id,
                        }
                    })
                })
            })
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
            self.process_spikes(ctx);
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
                    let projections = self
                        .nid_to_projections
                        .get_mut(&event.syn_coord.pre_syn_nid)
                        .unwrap();

                    let projection = &mut projections[event.syn_coord.projection_idx];

                    let synapse = &mut projection.synapses[event.syn_coord.synapse_idx];

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

    fn process_sc_eligible_spike_coincidences(
        ctx: &TickContext,
        spike_coincidences: impl Iterator<Item = SpikeCoincidence>,
        post_syn_nid: usize,
        nid_to_projections: &mut HashMap<usize, Vec<Projection>>,
        plasticity_modulator: &mut Option<PlasticityModulator>,
        sc_mode: SCMode,
        sc_hash_buffer: &mut HashSet<u64>,
    ) {
        match sc_mode {
            SCMode::Off => {
                for spike_coincidence in spike_coincidences {
                    Self::process_spike_coincidence(
                        nid_to_projections,
                        plasticity_modulator,
                        ctx.t,
                        &spike_coincidence,
                    );
                }
            }
            SCMode::Single { threshold } => {
                for spike_coincidence in spike_coincidences {
                    let processed = Self::process_spike_coincidence(
                        nid_to_projections,
                        plasticity_modulator,
                        ctx.t,
                        &spike_coincidence,
                    );

                    if processed.is_sc_contrib(threshold) {
                        let mut hasher = DefaultHasher::new();
                        hasher.write_usize(spike_coincidence.syn_coord.pre_syn_nid);
                        hasher.write_usize(post_syn_nid);
                        sc_hash_buffer.insert(hasher.finish());
                    }
                }
            }
            SCMode::Multi { threshold } => {
                let mut hasher = DefaultHasher::new();
                for spike_coincidence in spike_coincidences {
                    let processed = Self::process_spike_coincidence(
                        nid_to_projections,
                        plasticity_modulator,
                        ctx.t,
                        &spike_coincidence,
                    );
                    if processed.is_sc_contrib(threshold) {
                        hasher.write_usize(spike_coincidence.syn_coord.pre_syn_nid);
                    }
                }
                hasher.write_usize(post_syn_nid);
                sc_hash_buffer.insert(hasher.finish());
            }
        }
    }

    fn check_for_spikes(&mut self, ctx: &TickContext) -> Vec<usize> {
        let mut spiking_nids = Vec::new();

        // externally triggered spikes
        for spike_trigger_nid in &ctx.spike_trigger_nids {
            if *spike_trigger_nid >= self.nid_start {
                let neuron_idx = spike_trigger_nid - self.nid_start;

                if neuron_idx < self.neurons.len() {
                    if let Some(para_neurons) = &mut self.para_neurons {
                        para_neurons[neuron_idx].spike(ctx.t, &self.neuron_params);
                    }

                    let spike_coincidences =
                        self.neurons[neuron_idx].spike(ctx.t, &self.neuron_params).0;

                    Self::process_sc_eligible_spike_coincidences(
                        ctx,
                        spike_coincidences,
                        *spike_trigger_nid,
                        &mut self.nid_to_projections,
                        &mut self.plasticity_modulator,
                        self.sc_mode,
                        &mut self.sc_hash_buffer,
                    );

                    spiking_nids.push(*spike_trigger_nid);
                }
            }
        }

        // internally triggered spikes
        for neuron_idx in self.neuron_indexes_to_check.drain(..) {
            let mut clear_spike_coincidences = false;
            {
                let spike = self.neurons[neuron_idx].check_spike(ctx.t, &self.neuron_params);

                if let Some(para_neurons) = &mut self.para_neurons {
                    // para spikes enabled
                    let post_syn_nid = self.nid_start + neuron_idx;
                    let para_spike =
                        para_neurons[neuron_idx].check_spike(ctx.t, &self.neuron_params);

                    if let Some(para_spike) = para_spike {
                        Self::process_sc_eligible_spike_coincidences(
                            ctx,
                            para_spike.0,
                            post_syn_nid,
                            &mut self.nid_to_projections,
                            &mut self.plasticity_modulator,
                            self.sc_mode,
                            &mut self.sc_hash_buffer,
                        );

                        clear_spike_coincidences = true;
                    }

                    if spike.is_some() {
                        clear_spike_coincidences = true;
                        spiking_nids.push(post_syn_nid);
                    }
                } else if let Some(spike) = spike {
                    // para spikes disabled
                    let post_syn_nid = self.nid_start + neuron_idx;

                    Self::process_sc_eligible_spike_coincidences(
                        ctx,
                        spike.0,
                        post_syn_nid,
                        &mut self.nid_to_projections,
                        &mut self.plasticity_modulator,
                        self.sc_mode,
                        &mut self.sc_hash_buffer,
                    );

                    spiking_nids.push(post_syn_nid);
                }
            }

            if clear_spike_coincidences {
                self.neurons[neuron_idx].clear_spike_coincidences();
                self.para_neurons.as_deref_mut().unwrap()[neuron_idx].clear_spike_coincidences();
            }
        }

        spiking_nids
    }

    fn process_spike_coincidence(
        nid_to_projections: &mut HashMap<usize, Vec<Projection>>,
        plasticity_modulator: &mut Option<PlasticityModulator>,
        t: usize,
        spike_coincidence: &SpikeCoincidence,
    ) -> ProcessSpikeCoincidenceResult {
        let projections = nid_to_projections
            .get_mut(&spike_coincidence.syn_coord.pre_syn_nid)
            .unwrap();

        let projection = &mut projections[spike_coincidence.syn_coord.projection_idx];

        let synapse = &mut projection.synapses[spike_coincidence.syn_coord.synapse_idx];
        let syn_weight_before = synapse.weight;

        if let Some(short_term_stdp_params) = &projection.prj_params.short_term_stdp_params {
            let stdp_value = util::compute_stdp(
                spike_coincidence.t_pre_minus_post,
                &short_term_stdp_params.stdp_params,
            );

            synapse.process_short_term_stdp(t, stdp_value, short_term_stdp_params.tau);
        }

        if let Some(long_term_stdp_params) = &projection.prj_params.long_term_stdp_params {
            let stdp_value =
                util::compute_stdp(spike_coincidence.t_pre_minus_post, long_term_stdp_params);

            if let Some(plasticity_modulator) = plasticity_modulator {
                plasticity_modulator.process_stdp_value(
                    t,
                    &spike_coincidence.syn_coord,
                    stdp_value,
                );
            } else {
                synapse.process_weight_change(stdp_value, &projection.prj_params.synapse_params)
            }
        }

        ProcessSpikeCoincidenceResult {
            syn_weight: syn_weight_before,
            max_weight: projection.prj_params.synapse_params.max_weight,
            weight_scale_factor: projection.prj_params.synapse_params.weight_scale_factor,
        }
    }

    fn process_transmission_events(&mut self, t: usize, synaptic_transmission_count: &mut usize) {
        *synaptic_transmission_count += self.transmission_buffer.next_batch_size();

        for transm_event in self.transmission_buffer.drain_and_advance() {
            let psp_result = self.neurons[transm_event.neuron_idx].apply_psp(
                t,
                transm_event.psp,
                &transm_event.syn_coord,
                &self.neuron_params,
            );

            let para_might_spike = if let Some(para_neurons) = &mut self.para_neurons {
                let para_psp_result = para_neurons[transm_event.neuron_idx].apply_psp(
                    t,
                    transm_event.para_psp,
                    &transm_event.syn_coord,
                    &self.neuron_params,
                );

                para_psp_result.might_spike
            } else {
                false
            };

            if psp_result.might_spike || para_might_spike {
                self.neuron_indexes_to_check.push(transm_event.neuron_idx);
            }

            if let Some(spike_coincidence) = psp_result.spike_coincidence {
                let next_to_last_pre_syn_spike_t = self.nid_to_projections
                    [&transm_event.syn_coord.pre_syn_nid][transm_event.syn_coord.projection_idx]
                    .next_to_last_pre_syn_spike_t;

                let last_post_syn_spike_t =
                    self.neurons[transm_event.neuron_idx].get_last_spike_t();

                let is_eligible =
                    if let (Some(next_to_last_pre_syn_spike_t), Some(last_post_syn_spike_t)) =
                        (next_to_last_pre_syn_spike_t, last_post_syn_spike_t)
                    {
                        let next_to_last_transm_t = next_to_last_pre_syn_spike_t
                            + self.nid_to_projections[&transm_event.syn_coord.pre_syn_nid]
                                [transm_event.syn_coord.projection_idx]
                                .synapses[transm_event.syn_coord.synapse_idx]
                                .conduction_delay as usize;

                        next_to_last_transm_t <= last_post_syn_spike_t
                    } else {
                        true
                    };

                if is_eligible {
                    Self::process_spike_coincidence(
                        &mut self.nid_to_projections,
                        &mut self.plasticity_modulator,
                        t,
                        &spike_coincidence,
                    );
                }
            }
        }
    }

    fn process_spike(&mut self, spike_t: usize, spiking_nid: usize) {
        if let Some(projections) = self.nid_to_projections.get_mut(&spiking_nid) {
            for (projection_idx, projection) in projections.iter_mut().enumerate() {
                projection.next_to_last_pre_syn_spike_t = projection.last_pre_syn_spike_t;
                projection.last_pre_syn_spike_t = Some(spike_t);

                let stp_value = projection.stp.on_pre_syn_spike_get_value(spike_t);

                let short_term_stdp_tau = projection
                    .prj_params
                    .short_term_stdp_params
                    .as_ref()
                    .map(|short_term_stdp_params| short_term_stdp_params.tau)
                    .unwrap_or(1.0); // if none, default won't have any effect

                for (synapse_idx, synapse) in projection.synapses.iter_mut().enumerate() {
                    let psp_result = synapse.process_pre_syn_spike_get_psp(
                        spike_t,
                        stp_value,
                        &projection.prj_params.synapse_params,
                        short_term_stdp_tau,
                    );

                    let syn_coord = SynapseCoordinate {
                        pre_syn_nid: spiking_nid,
                        projection_idx,
                        synapse_idx,
                    };

                    let transmission_event = TransmissionEvent {
                        syn_coord,
                        neuron_idx: synapse.neuron_idx,
                        psp: psp_result.psp,
                        para_psp: psp_result.para_psp,
                    };

                    self.transmission_buffer
                        .push_at_offset(synapse.conduction_delay as usize - 1, transmission_event);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        params::{LayerParams, TechnicalParams},
        types::tests::HashSet,
    };

    use super::*;
    use itertools::assert_equal;
    use rand::SeedableRng;

    #[test]
    fn conduction_delay() {
        let mut conn_params = LayerConnectionParams {
            from_layer_id: 5,
            to_layer_id: 3,
            projection_params: ProjectionParams::default(),
            smooth_connect_probability: false,
            connect_width: f64::INFINITY,
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
            smooth_connect_probability: false,
            connect_width: f64::INFINITY,
            initial_syn_weight: InitialSynWeight::Constant(0.5),
            conduction_delay_max_random_part: 0,
            conduction_delay_position_distance_scale_factor: 2.0,
            conduction_delay_add_on: 0,
            allow_self_innervation: true,
        };

        let instance_params = InstanceParams {
            layers,
            layer_connections: vec![connection_params],
            position_dim: 1,
            hyper_sphere: false,
            technical_params: TechnicalParams::default(),
        };

        let partitions = create_partitions(3, 2, &instance_params);

        assert_eq!(partitions.len(), 2);

        assert_eq!(partitions[0].nid_start, 67);
        assert_eq!(partitions[1].nid_start, 134);

        assert_eq!(partitions[0].get_num_neurons(), 33);
        assert_eq!(partitions[1].get_num_neurons(), 16);

        assert!(partitions[0].nid_to_projections.is_empty());
        assert_eq!(partitions[1].nid_to_projections.len(), 100);

        assert_eq!(partitions[1].nid_to_projections[&0][0].synapses.len(), 16);
        assert_eq!(
            partitions[1].nid_to_projections[&0][0].synapses[0].neuron_idx,
            0
        );
        assert_eq!(
            partitions[1].nid_to_projections[&0][0].synapses[0].conduction_delay,
            2
        );

        assert_eq!(
            partitions[1].nid_to_projections[&0][0].synapses[15].neuron_idx,
            15
        );
        assert_eq!(
            partitions[1].nid_to_projections[&0][0].synapses[15].conduction_delay,
            3
        );
    }

    #[test]
    fn randomized_weights() {
        let layer = LayerParams {
            num_neurons: 100,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
            use_para_spikes: false,
        };

        let conn_params = LayerConnectionParams {
            from_layer_id: 0,
            to_layer_id: 0,
            projection_params: ProjectionParams::default(),
            smooth_connect_probability: false,
            connect_width: f64::INFINITY,
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
        assert_eq!(partitions[0].nid_to_projections.len(), 100);

        for (_, projections) in &partitions[0].nid_to_projections {
            assert_eq!(projections.len(), 1);
            let projection = &projections[0];
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
    fn partition_creation() {
        let mut params = InstanceParams::default();
        params.position_dim = 1;

        let layer_0 = LayerParams {
            num_neurons: 5,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
            use_para_spikes: false,
        };

        let layer_1 = LayerParams {
            num_neurons: 100,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
            use_para_spikes: false,
        };

        let layer_2 = LayerParams {
            num_neurons: 10,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
            use_para_spikes: false,
        };

        params.layers.push(layer_0);
        params.layers.push(layer_1);
        params.layers.push(layer_2);

        let connection_params = LayerConnectionParams {
            from_layer_id: 1,
            to_layer_id: 2,
            projection_params: ProjectionParams::default(),
            smooth_connect_probability: false,
            connect_width: f64::INFINITY,
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

        assert!(partitions[0].nid_to_projections.is_empty());
        assert!(partitions[1].nid_to_projections.is_empty());
        assert_eq!(partitions[2].nid_to_projections.len(), 100);

        for nid in 5..105 {
            assert!(partitions[2].nid_to_projections.contains_key(&nid));
            assert_equal(
                partitions[2].nid_to_projections[&nid][0]
                    .synapses
                    .iter()
                    .map(|synapse| synapse.neuron_idx),
                0..10,
            );
        }

        let synapses_for_nid_65 = &partitions[2].nid_to_projections[&(65)][0].synapses;

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

        assert_eq!(partitions[2].nid_to_projections.len(), 100);

        for nid in 5..105 {
            assert!(partitions[2].nid_to_projections.contains_key(&nid));
        }

        assert_equal(
            partitions[2].nid_to_projections[&5][0]
                .synapses
                .iter()
                .map(|synapse| synapse.neuron_idx),
            0..2,
        );

        assert_equal(
            partitions[2].nid_to_projections[&104][0]
                .synapses
                .iter()
                .map(|synapse| synapse.neuron_idx),
            8..10,
        );

        assert_equal(
            partitions[2].nid_to_projections[&(5 + 60)][0]
                .synapses
                .iter()
                .map(|synapse| synapse.neuron_idx),
            4..8,
        );

        // sparse connection
        params.layer_connections[0].connect_width = 0.25;
        let partitions = create_partitions(1, 0, &params);

        assert_eq!(partitions[2].nid_to_projections.len(), 100);

        for nid in 5..105 {
            assert!(partitions[2].nid_to_projections.contains_key(&nid));

            // the below boundaries are chosen intuitively. Might not work with different seed or
            // generator
            assert!(partitions[2].nid_to_projections[&nid][0].synapses.len() > 1);
            assert!(partitions[2].nid_to_projections[&nid][0].synapses.len() < 5);
        }
    }

    #[test]
    fn zero_connect_width() {
        let mut params = InstanceParams::default();
        params.position_dim = 1;
        let mut layer = LayerParams::default();
        layer.num_neurons = 5;
        params.layers.push(layer.clone());
        layer.num_neurons = 3;
        params.layers.push(layer);

        let mut connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);
        connection.connect_width = 0.0;

        params.layer_connections.push(connection);

        let partitions = create_partitions(1, 0, &params);

        assert_eq!(partitions[1].nid_to_projections.len(), 3);

        assert_eq!(partitions[1].nid_to_projections[&0][0].synapses.len(), 1);
        assert_eq!(
            partitions[1].nid_to_projections[&0][0].synapses[0].neuron_idx,
            0
        );

        assert_eq!(partitions[1].nid_to_projections[&2][0].synapses.len(), 1);
        assert_eq!(
            partitions[1].nid_to_projections[&2][0].synapses[0].neuron_idx,
            1
        );

        assert_eq!(partitions[1].nid_to_projections[&4][0].synapses.len(), 1);
        assert_eq!(
            partitions[1].nid_to_projections[&4][0].synapses[0].neuron_idx,
            2
        );
    }
}
