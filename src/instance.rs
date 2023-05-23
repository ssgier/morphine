use crate::api::SCMode;
use crate::params;
use crate::params::InstanceParams;
use crate::partition;
use crate::partition::Partition;
use crate::partition::PartitionGroupResult;
use crate::partition::PartitionStateSnapshot;
use crate::partition::Request;
use crate::partition::TickContext;
use crate::state_snapshot::StateSnapshot;
use crate::types::HashMap;
use bus::Bus;
use core_affinity::CoreId;
use itertools::Itertools;
use num_cpus;
use simple_error::SimpleResult;
use simple_error::{try_with, SimpleError};
use std::collections::HashSet;
use std::sync::mpsc::channel as mpsc_channel;
use std::sync::mpsc::Receiver as MpscReceiver;
use std::thread;
use std::thread::JoinHandle;
use std::usize;

pub fn create_instance(params: InstanceParams) -> Result<Instance, SimpleError> {
    try_with!(
        params::validate_instance_params(&params),
        "invalid instance parameters"
    );

    let mut broadcast_tx = Bus::new(1);
    let (partition_result_tx, partition_result_rx) = mpsc_channel();
    let (partition_snapshots_tx, partition_snapshots_rx) = mpsc_channel();
    let (partition_ack_tx, partition_ack_rx) = mpsc_channel();
    let (partition_sc_hashes_tx, partition_sc_hashes_rx) = mpsc_channel();

    let num_threads = get_num_threads(&params);
    let mut join_handles = Vec::new();

    for thread_id in 0..num_threads {
        let broadcast_rx = broadcast_tx.add_rx();
        let partition_result_tx = partition_result_tx.clone();
        let partition_snapshots_tx = partition_snapshots_tx.clone();
        let partition_ack_tx = partition_ack_tx.clone();
        let partition_sc_hashes_tx = partition_sc_hashes_tx.clone();
        let params = params.clone();

        join_handles.push(thread::spawn(move || {
            if params.technical_params.pin_threads {
                let core_id = CoreId { id: thread_id };
                core_affinity::set_for_current(core_id);
            }

            let mut partitions = partition::create_partitions(num_threads, thread_id, &params);
            Partition::run(
                &mut partitions,
                broadcast_rx,
                partition_result_tx,
                partition_snapshots_tx,
                partition_ack_tx,
                partition_sc_hashes_tx,
            );
        }));
    }

    let nid_to_out_channel_id = create_out_nid_channel_mapping(&params);

    let broadcast_tx = Some(broadcast_tx);

    let num_in_channels = if let Some(layer) = params.layers.first() {
        layer.num_neurons
    } else {
        0
    };

    let num_neurons = params.layers.iter().map(|layer| layer.num_neurons).sum();

    Ok(Instance {
        num_in_channels,
        num_neurons,
        out_channel_id_to_nid: nid_to_out_channel_id
            .iter()
            .map(|(nid, channel)| (*channel, *nid))
            .collect(),
        nid_to_out_channel_id,
        spiking_nid_buffer: Vec::new(),
        broadcast_tx,
        partition_result_rx,
        partition_snapshots_rx,
        partition_ack_rx,
        partition_sc_hashes_rx,
        num_partitions: num_threads,
        last_tick_period: 0,
        join_handles,
    })
}

fn aggregate_state_snapshot(
    partition_group_snapshots: Vec<Vec<PartitionStateSnapshot>>,
) -> StateSnapshot {
    let mut neuron_states = Vec::new();
    let mut synapse_states = Vec::new();

    let partition_snapshots_ordered = partition_group_snapshots
        .into_iter()
        .flatten()
        .sorted_by_key(|partition_snapshot| partition_snapshot.nid_start);

    for mut partition_snapshot in partition_snapshots_ordered {
        neuron_states.append(&mut partition_snapshot.neuron_states);
        synapse_states.append(&mut partition_snapshot.synapse_states);
    }

    StateSnapshot {
        neuron_states,
        synapse_states,
    }
}

fn get_num_threads(params: &InstanceParams) -> usize {
    params
        .technical_params
        .num_threads
        .unwrap_or_else(num_cpus::get)
}

fn create_out_nid_channel_mapping(params: &InstanceParams) -> HashMap<usize, usize> {
    let mut result = HashMap::default();
    let mut non_output_neuron_count = 0;

    if !params.layers.is_empty() {
        for i in 0..(params.layers.len() - 1) {
            non_output_neuron_count += params.layers[i].num_neurons;
        }

        for out_channel_id in 0..params.layers.last().unwrap().num_neurons {
            result.insert(out_channel_id + non_output_neuron_count, out_channel_id);
        }
    }

    result
}

#[derive(Debug, Clone)]
pub struct TickInput {
    pub spiking_in_channel_ids: Vec<usize>,
    pub force_spiking_out_channel_ids: Vec<usize>,
    pub force_spiking_nids: Vec<usize>,
    pub reward: f32,
}

impl Default for TickInput {
    fn default() -> Self {
        Self::new()
    }
}

impl TickInput {
    pub fn new() -> Self {
        EMPTY_TICK_INPUT.clone()
    }

    pub fn from_spiking_in_channel_ids(spiking_in_channel_ids: &[usize]) -> Self {
        let mut result = EMPTY_TICK_INPUT.clone();
        result
            .spiking_in_channel_ids
            .extend_from_slice(spiking_in_channel_ids);
        result
    }

    pub fn from_reward(reward: f32) -> Self {
        let mut result = EMPTY_TICK_INPUT.clone();
        result.reward = reward;
        result
    }

    pub fn reset(&mut self) {
        *self = EMPTY_TICK_INPUT.clone();
    }
}

#[derive(Debug)]
pub struct TickResult {
    pub t: usize,
    pub spiking_out_channel_ids: Vec<usize>,
    pub spiking_nids: Vec<usize>,
    pub synaptic_transmission_count: usize,
}

pub struct Instance {
    num_in_channels: usize,
    num_neurons: usize,
    nid_to_out_channel_id: HashMap<usize, usize>,
    out_channel_id_to_nid: HashMap<usize, usize>,
    spiking_nid_buffer: Vec<usize>,
    broadcast_tx: Option<Bus<Request>>,
    partition_result_rx: MpscReceiver<PartitionGroupResult>,
    partition_snapshots_rx: MpscReceiver<Vec<PartitionStateSnapshot>>,
    partition_ack_rx: MpscReceiver<()>,
    partition_sc_hashes_rx: MpscReceiver<HashSet<u64>>,
    num_partitions: usize,
    last_tick_period: usize,
    join_handles: Vec<JoinHandle<()>>,
}

static EMPTY_TICK_INPUT: TickInput = TickInput {
    spiking_in_channel_ids: Vec::new(),
    force_spiking_out_channel_ids: Vec::new(),
    force_spiking_nids: Vec::new(),
    reward: 0.0,
};

impl Instance {
    pub fn get_num_neurons(&self) -> usize {
        self.num_neurons
    }

    pub fn get_num_in_channels(&self) -> usize {
        self.num_in_channels
    }

    pub fn get_num_out_channels(&self) -> usize {
        self.nid_to_out_channel_id.len()
    }

    pub fn tick(&mut self, tick_input: &TickInput) -> SimpleResult<TickResult> {
        self.last_tick_period += 1;

        self.validate_tick_input(tick_input)?;

        let mut spike_trigger_nids: Vec<usize> = tick_input.spiking_in_channel_ids.to_vec();
        spike_trigger_nids.extend_from_slice(&tick_input.force_spiking_nids);
        spike_trigger_nids.extend(
            tick_input
                .force_spiking_out_channel_ids
                .iter()
                .map(|channel_id| self.out_channel_id_to_nid.get(channel_id).unwrap()),
        );

        let spiked_nids: Vec<_> = self.spiking_nid_buffer.drain(0..).collect();

        let dopamine_amount = tick_input.reward; // to be revised. There might be an indirection via dopaminergic neurons

        let t = self.last_tick_period;
        let ctx = Request::Tick(TickContext {
            t,
            spike_trigger_nids,
            spiked_nids,
            dopamine_amount,
        });

        self.broadcast_tx.as_mut().unwrap().broadcast(ctx);

        let mut partition_group_results = Vec::new();

        for _ in 0..self.num_partitions {
            partition_group_results.push(self.partition_result_rx.recv().unwrap());
        }

        let mut synaptic_transmission_count = 0;
        for partition_group_result in &partition_group_results {
            synaptic_transmission_count += partition_group_result.synaptic_transmission_count;
            self.spiking_nid_buffer
                .extend(&partition_group_result.spiking_nids);
        }

        self.spiking_nid_buffer.sort();

        let out_spiking_channel_ids: Vec<usize> = self
            .spiking_nid_buffer
            .iter()
            .filter_map(|spiking_nid| self.nid_to_out_channel_id.get(spiking_nid).copied())
            .collect();

        Ok(TickResult {
            spiking_out_channel_ids: out_spiking_channel_ids,
            spiking_nids: self.spiking_nid_buffer.clone(),
            synaptic_transmission_count,
            t,
        })
    }

    pub fn tick_no_input(&mut self) -> TickResult {
        self.tick(&EMPTY_TICK_INPUT).unwrap()
    }

    pub fn tick_no_input_until_inclusive(&mut self, t: usize) {
        while self.get_last_tick_period() < t {
            self.tick_no_input();
        }
    }

    pub fn tick_no_input_until_exclusive(&mut self, t: usize) {
        while self.get_next_tick_period() < t {
            self.tick_no_input();
        }
    }

    pub fn get_last_tick_period(&self) -> usize {
        self.last_tick_period
    }

    pub fn get_next_tick_period(&self) -> usize {
        self.last_tick_period + 1
    }

    pub fn extract_state_snapshot(&mut self) -> StateSnapshot {
        self.broadcast_tx
            .as_mut()
            .unwrap()
            .broadcast(Request::ExtractStateSnapshot {
                t: self.last_tick_period,
            });

        let mut partition_group_snapshots = Vec::new();

        for _ in 0..self.num_partitions {
            partition_group_snapshots.push(self.partition_snapshots_rx.recv().unwrap());
        }

        aggregate_state_snapshot(partition_group_snapshots)
    }

    pub fn reset_ephemeral_state(&mut self) {
        self.spiking_nid_buffer.clear();

        self.broadcast_tx
            .as_mut()
            .unwrap()
            .broadcast(Request::ResetEphemeralState {
                t: self.last_tick_period,
            });

        for _ in 0..self.num_partitions {
            self.partition_ack_rx.recv().unwrap();
        }
    }

    pub fn set_sc_mode(&mut self, sc_mode: SCMode) {
        self.broadcast_tx
            .as_mut()
            .unwrap()
            .broadcast(Request::SetSCMode(sc_mode))
    }

    pub fn flush_sc_hashes(&mut self) -> HashSet<u64> {
        self.broadcast_tx
            .as_mut()
            .unwrap()
            .broadcast(Request::FlushSCHashes);

        (0..self.num_partitions)
            .flat_map(|_ignored| self.partition_sc_hashes_rx.recv().unwrap().into_iter())
            .collect()
    }

    fn validate_tick_input(&self, tick_input: &TickInput) -> SimpleResult<()> {
        for in_channel_id in &tick_input.spiking_in_channel_ids {
            if *in_channel_id >= self.num_in_channels {
                return Err(SimpleError::new(format!(
                    "Invalid input channel id: {}",
                    in_channel_id
                )));
            }
        }

        for nid in &tick_input.force_spiking_nids {
            if *nid >= self.num_neurons {
                return Err(SimpleError::new(format!(
                    "Invalid neuron id for forced spike: {}",
                    nid
                )));
            }
        }

        for out_channel_id in &tick_input.force_spiking_out_channel_ids {
            if *out_channel_id >= self.get_num_out_channels() {
                return Err(SimpleError::new(format!(
                    "Invalid output channel id for forced spike: {}",
                    out_channel_id
                )));
            }
        }

        Ok(())
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        drop(self.broadcast_tx.take()); // signals the worker threads to exit the loop

        self.join_handles.drain(..).for_each(|join_handle| {
            join_handle.join().ok();
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::params::{
        InitialSynWeight, LayerConnectionParams, PlasticityModulationParams, ShortTermStdpParams,
        StdpParams, StpParams,
    };
    use crate::params::{LayerParams, NeuronParams};
    use crate::partition::{PartitionGroupResult, PartitionStateSnapshot};
    use crate::state_snapshot::{NeuronState, SynapseState};
    use crate::types::HashMap;

    use super::*;
    use float_cmp::assert_approx_eq;
    use itertools::assert_equal;
    use std::thread;

    #[test]
    fn simple_round_trip() {
        let num_partitions = 2;

        let broadcast_tx = Some(Bus::new(1));
        let (partition_result_tx, partition_result_rx) = mpsc_channel();
        let (_, partition_snapshots_rx) = mpsc_channel();
        let (_, partition_ack_rx) = mpsc_channel();
        let (_, partition_sc_hashes_rx) = mpsc_channel();

        let spiking_in_channel_id = 3;
        let channel_mapped_nid = 3;
        let spiking_nid = 10;
        let spiking_out_channel_id = 11;

        let nid_to_out_channel_id = HashMap::from_iter([(spiking_nid, spiking_out_channel_id)]);
        let out_channel_id_to_nid = HashMap::from_iter([(spiking_out_channel_id, spiking_nid)]);

        let mut instance = Instance {
            num_in_channels: 10,
            num_neurons: 20,
            out_channel_id_to_nid,
            nid_to_out_channel_id,
            spiking_nid_buffer: Vec::new(),
            broadcast_tx,
            partition_result_rx,
            partition_snapshots_rx,
            partition_ack_rx,
            partition_sc_hashes_rx,
            num_partitions,
            last_tick_period: 0,
            join_handles: Vec::new(),
        };

        let join_handles: Vec<_> = (0..num_partitions)
            .into_iter()
            .map(|partition_id| {
                let mut broadcast_rx = instance.broadcast_tx.as_mut().unwrap().add_rx();
                let partition_result_tx = partition_result_tx.clone();
                thread::spawn(move || {
                    // first cycle
                    let ctx_1st_cycle = broadcast_rx.recv().unwrap();

                    let spiking_nids = if partition_id == 1 {
                        vec![spiking_nid]
                    } else {
                        Vec::new()
                    };

                    partition_result_tx
                        .send(PartitionGroupResult {
                            spiking_nids,
                            synaptic_transmission_count: 0,
                        })
                        .unwrap();

                    //second cycle
                    let ctx_2nd_cycle = broadcast_rx.recv().unwrap();

                    partition_result_tx
                        .send(PartitionGroupResult {
                            spiking_nids: Vec::new(),
                            synaptic_transmission_count: 0,
                        })
                        .unwrap();

                    (ctx_1st_cycle, ctx_2nd_cycle)
                })
            })
            .collect();

        let mut tick_input = TickInput::new();
        tick_input
            .spiking_in_channel_ids
            .push(spiking_in_channel_id);

        // first cycle
        let tick_result = instance.tick(&tick_input).unwrap();
        assert_eq!(
            tick_result.spiking_out_channel_ids,
            [spiking_out_channel_id]
        );
        assert_eq!(instance.spiking_nid_buffer, [spiking_nid]);
        assert_eq!(instance.get_last_tick_period(), 1);

        // second cycle
        let tick_result = instance.tick_no_input();
        assert!(tick_result.spiking_out_channel_ids.is_empty());
        assert!(instance.spiking_nid_buffer.is_empty());
        assert_eq!(instance.get_last_tick_period(), 2);

        for join_handle in join_handles {
            if let (Request::Tick(ctx_1st), Request::Tick(ctx_2nd)) = join_handle.join().unwrap() {
                assert_eq!(ctx_1st.spike_trigger_nids, [channel_mapped_nid]);
                assert_eq!(ctx_1st.spiked_nids, [] as [usize; 0]);
                assert_eq!(ctx_2nd.spike_trigger_nids, [] as [usize; 0]);
                assert_eq!(ctx_2nd.spiked_nids, [spiking_nid]);
            } else {
                panic!();
            }
        }
    }

    #[test]
    fn echo_instance() {
        let num_partitions = 1;

        let broadcast_tx = Some(Bus::new(1));
        let (partition_result_tx, partition_result_rx) = mpsc_channel();
        let (_, partition_snapshots_rx) = mpsc_channel();
        let (_, partition_ack_rx) = mpsc_channel();
        let (_, partition_sc_hashes_rx) = mpsc_channel();
        let mut out_nid_channel_mapping = HashMap::default();

        for i in 0..10 {
            out_nid_channel_mapping.insert(i + 10, i);
        }

        let mut instance = Instance {
            num_in_channels: 10,
            num_neurons: 20,
            out_channel_id_to_nid: out_nid_channel_mapping
                .iter()
                .map(|(nid, channel)| (*channel, *nid))
                .collect(),
            nid_to_out_channel_id: out_nid_channel_mapping,
            spiking_nid_buffer: Vec::new(),
            broadcast_tx,
            partition_result_rx,
            partition_snapshots_rx,
            partition_ack_rx,
            partition_sc_hashes_rx,
            num_partitions,
            last_tick_period: 0,
            join_handles: Vec::new(),
        };

        let mut broadcast_rx = instance.broadcast_tx.as_mut().unwrap().add_rx();

        let join_handle = thread::spawn(move || {
            while let Ok(Request::Tick(ctx)) = broadcast_rx.recv() {
                let spiking_nids = ctx
                    .spike_trigger_nids
                    .into_iter()
                    .map(|id| id + 10)
                    .collect();
                partition_result_tx
                    .send(PartitionGroupResult {
                        spiking_nids,
                        synaptic_transmission_count: 0,
                    })
                    .unwrap();
            }
        });

        let inputs = [vec![1, 4, 5], vec![1, 3], vec![], vec![9]];

        let mut tick_input = TickInput::new();
        for input in inputs {
            tick_input.spiking_in_channel_ids = input.clone();
            let tick_result = instance.tick(&tick_input).unwrap();
            assert_eq!(tick_result.spiking_out_channel_ids, input);
        }

        drop(instance); // this drops the channel so the partition thread stops

        join_handle.join().unwrap();
    }

    #[test]
    fn out_channel_mapping() {
        let mut params = InstanceParams::default();

        params.layers = vec![
            LayerParams {
                num_neurons: 5,
                neuron_params: NeuronParams::default(),
                plasticity_modulation_params: None,
            },
            LayerParams {
                num_neurons: 3,
                neuron_params: NeuronParams::default(),
                plasticity_modulation_params: None,
            },
            LayerParams {
                num_neurons: 2,
                neuron_params: NeuronParams::default(),
                plasticity_modulation_params: None,
            },
        ];

        let channel_mapping = create_out_nid_channel_mapping(&params);

        assert_eq!(channel_mapping, HashMap::from_iter([(8, 0), (9, 1)]));
    }

    #[test]
    fn state_snapshot_aggregation() {
        let partition_snapshot_0 = PartitionStateSnapshot {
            nid_start: 4,
            neuron_states: vec![
                NeuronState {
                    voltage: 4.0,
                    threshold: 4.0,
                    is_refractory: false,
                },
                NeuronState {
                    voltage: 5.0,
                    threshold: 5.0,
                    is_refractory: false,
                },
            ],
            synapse_states: vec![SynapseState {
                projection_id: 0,
                pre_syn_nid: 0,
                post_syn_nid: 5,
                conduction_delay: 3,
                weight: 0.2,
                short_term_stdp_offset: 0.2,
            }],
        };

        let partition_snapshot_1 = PartitionStateSnapshot {
            nid_start: 1,
            neuron_states: vec![
                NeuronState {
                    voltage: 1.0,
                    threshold: 1.0,
                    is_refractory: false,
                },
                NeuronState {
                    voltage: 2.0,
                    threshold: 2.0,
                    is_refractory: false,
                },
                NeuronState {
                    voltage: 3.0,
                    threshold: 3.0,
                    is_refractory: false,
                },
            ],
            synapse_states: vec![
                SynapseState {
                    projection_id: 0,
                    pre_syn_nid: 0,
                    post_syn_nid: 1,
                    conduction_delay: 1,
                    weight: 0.2,
                    short_term_stdp_offset: 0.2,
                },
                SynapseState {
                    projection_id: 0,
                    pre_syn_nid: 0,
                    post_syn_nid: 2,
                    conduction_delay: 2,
                    weight: 0.3,
                    short_term_stdp_offset: 0.3,
                },
            ],
        };

        let partition_snapshot_2 = PartitionStateSnapshot {
            nid_start: 0,
            neuron_states: vec![NeuronState {
                voltage: 0.0,
                threshold: 0.5,
                is_refractory: false,
            }],
            synapse_states: Vec::new(),
        };

        let group_snapshots_0 = vec![partition_snapshot_0, partition_snapshot_2];
        let group_snapshots_1 = vec![partition_snapshot_1];
        let group_snapshots = vec![group_snapshots_0, group_snapshots_1];

        let state_snapshot = aggregate_state_snapshot(group_snapshots);

        for (index, snapshot) in state_snapshot.neuron_states.iter().enumerate() {
            let expected_value = index as f32;
            assert_approx_eq!(f32, snapshot.voltage, expected_value);

            if index == 0 {
                assert_approx_eq!(f32, snapshot.threshold, 0.5);
            } else {
                assert_approx_eq!(f32, snapshot.threshold, expected_value);
            }
        }

        assert_eq!(state_snapshot.synapse_states.len(), 3);

        assert_eq!(state_snapshot.synapse_states[0].pre_syn_nid, 0);
        assert_eq!(state_snapshot.synapse_states[0].post_syn_nid, 1);
        assert_eq!(state_snapshot.synapse_states[0].conduction_delay, 1);
        assert_approx_eq!(f32, state_snapshot.synapse_states[0].weight, 0.2);
        assert_approx_eq!(
            f32,
            state_snapshot.synapse_states[0].short_term_stdp_offset,
            0.2
        );

        assert_eq!(state_snapshot.synapse_states[1].pre_syn_nid, 0);
        assert_eq!(state_snapshot.synapse_states[1].post_syn_nid, 2);
        assert_eq!(state_snapshot.synapse_states[1].conduction_delay, 2);
        assert_approx_eq!(f32, state_snapshot.synapse_states[1].weight, 0.3);
        assert_approx_eq!(
            f32,
            state_snapshot.synapse_states[1].short_term_stdp_offset,
            0.3
        );

        assert_eq!(state_snapshot.synapse_states[2].pre_syn_nid, 0);
        assert_eq!(state_snapshot.synapse_states[2].post_syn_nid, 5);
        assert_eq!(state_snapshot.synapse_states[2].conduction_delay, 3);
        assert_approx_eq!(f32, state_snapshot.synapse_states[2].weight, 0.2);
        assert_approx_eq!(
            f32,
            state_snapshot.synapse_states[2].short_term_stdp_offset,
            0.2
        );
    }

    #[test]
    fn forced_spikes() {
        let mut params = InstanceParams::default();
        let mut layer = LayerParams::default();
        layer.num_neurons = 10;
        params.layers.push(layer.clone());
        layer.num_neurons = 5;
        params.layers.push(layer);

        let mut instance = create_instance(params).unwrap();

        let mut tick_input = TickInput::new();
        tick_input.spiking_in_channel_ids.push(3);
        tick_input.force_spiking_nids = vec![3, 4, 12];
        tick_input.force_spiking_out_channel_ids.push(3);

        let tick_result = instance.tick(&tick_input).unwrap();

        assert_equal(tick_result.spiking_out_channel_ids, [2, 3]);
        assert_equal(tick_result.spiking_nids, [3, 3, 4, 12, 13]);
    }

    #[test]
    fn invalid_tick_input() {
        let mut params = InstanceParams::default();
        let mut layer = LayerParams::default();
        layer.num_neurons = 10;
        params.layers.push(layer.clone());
        layer.num_neurons = 5;
        params.layers.push(layer);

        let mut instance = create_instance(params).unwrap();

        let result = instance.tick(&TickInput::from_spiking_in_channel_ids(&[10]));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().as_str(), "Invalid input channel id: 10");

        let mut tick_input = TickInput::new();
        tick_input.force_spiking_nids.push(15);

        let result = instance.tick(&tick_input);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().as_str(),
            "Invalid neuron id for forced spike: 15"
        );

        let mut tick_input = TickInput::new();
        tick_input.force_spiking_out_channel_ids.push(5);

        let result = instance.tick(&tick_input);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().as_str(),
            "Invalid output channel id for forced spike: 5"
        );
    }

    #[test]
    fn ephemeral_state_reset() {
        let mut params = InstanceParams::default();

        let mut layer = LayerParams::default();

        layer.num_neurons = 1;
        params.layers.push(layer.clone());
        layer.neuron_params.refractory_period = 10;
        layer.neuron_params.adaptation_threshold = 2.0;
        layer.neuron_params.reset_voltage = 0.8;
        let mut plasticity_modulation_params = PlasticityModulationParams::default();
        plasticity_modulation_params.dopamine_conflation_period = 100;
        plasticity_modulation_params.dopamine_flush_period = 200;
        layer.plasticity_modulation_params = Some(plasticity_modulation_params);
        params.layers.push(layer);

        let mut last_layer = LayerParams::default();
        last_layer.num_neurons = 1;
        params.layers.push(last_layer);

        let mut layer_connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);

        layer_connection.projection_params.stp_params = StpParams::Depression {
            tau: 500.0,
            p0: 0.5,
            factor: 0.5,
        };

        layer_connection.projection_params.short_term_stdp_params = Some(ShortTermStdpParams {
            tau: 500.0,
            stdp_params: StdpParams::default(),
        });

        layer_connection.projection_params.long_term_stdp_params = Some(StdpParams::default());
        layer_connection.initial_syn_weight = InitialSynWeight::Constant(0.5);
        layer_connection.projection_params.synapse_params.max_weight = 1.0;
        layer_connection.conduction_delay_add_on = 1;

        params.layer_connections.push(layer_connection);

        let mut last_layer_connection = LayerConnectionParams::defaults_for_layer_ids(1, 2);
        last_layer_connection.initial_syn_weight = InitialSynWeight::Constant(1.0);
        params.layer_connections.push(last_layer_connection);

        let mut instance = create_instance(params).unwrap();

        let mut tick_input = TickInput::default();
        tick_input.force_spiking_nids = vec![0];
        instance.tick(&tick_input).unwrap();
        instance.tick_no_input();
        tick_input.force_spiking_nids = vec![1];
        instance.tick(&tick_input).unwrap();

        instance.reset_ephemeral_state();
        let state_snapshot = instance.extract_state_snapshot();
        assert!(!state_snapshot.neuron_states[1].is_refractory);
        assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.0);
        assert_approx_eq!(f32, state_snapshot.neuron_states[1].threshold, 1.0);
        assert_approx_eq!(
            f32,
            state_snapshot.synapse_states[0].short_term_stdp_offset,
            0.0
        );

        tick_input.reset();
        tick_input.reward = 1.0;
        instance.tick(&tick_input).unwrap();
        instance.tick_no_input_until_inclusive(210);

        tick_input.reset();
        tick_input.force_spiking_nids = vec![0];
        instance.tick(&tick_input).unwrap();
        instance.tick_no_input();
        instance.tick_no_input();
        let state_snapshot = instance.extract_state_snapshot();

        assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.25);

        tick_input.reset();
        tick_input.force_spiking_nids = vec![0];
        instance.tick(&tick_input).unwrap();
        instance.tick_no_input();

        // check that the transmission buffer gets cleared as well
        instance.reset_ephemeral_state();
        let state_snapshot = instance.extract_state_snapshot();
        assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.0);

        tick_input.reset();
        tick_input.force_spiking_nids = vec![1];
        instance.tick(&tick_input).unwrap();
        tick_input.reset();
        instance.reset_ephemeral_state();
        let tick_result = instance.tick(&tick_input).unwrap();
        assert!(tick_result.spiking_nids.is_empty());
    }
}
