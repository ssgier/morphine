use crate::params;
use crate::params::InstanceParams;
use crate::partition;
use crate::partition::Partition;
use crate::partition::PartitionGroupResult;
use crate::partition::TickContext;
use crate::state_snapshot::StateSnapshot;
use crate::types::HashMap;
use bus::Bus;
use core_affinity::CoreId;
use itertools::Itertools;
use num_cpus;
use simple_error::{try_with, SimpleError};
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

    let num_threads = get_num_threads(&params);
    let mut join_handles = Vec::new();

    for thread_id in 0..num_threads {
        let broadcast_rx = broadcast_tx.add_rx();
        let partition_result_tx = partition_result_tx.clone();
        let params = params.clone();

        join_handles.push(thread::spawn(move || {
            if params.technical_params.pin_threads {
                let core_id = CoreId { id: thread_id };
                core_affinity::set_for_current(core_id);
            }

            let mut partitions = partition::create_partitions(num_threads, thread_id, &params);
            Partition::run(&mut partitions, broadcast_rx, partition_result_tx);
        }));
    }

    let out_nid_channel_mapping = create_out_nid_channel_mapping(&params);

    let broadcast_tx = Some(broadcast_tx);

    Ok(Instance {
        out_nid_channel_mapping,
        spiking_nid_buffer: Vec::new(),
        broadcast_tx,
        partition_result_rx,
        num_partitions: num_threads,
        tick_period: 0,
        join_handles,
    })
}

fn get_num_threads(params: &InstanceParams) -> usize {
    params
        .technical_params
        .num_threads
        .unwrap_or_else(|| num_cpus::get())
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

#[derive(Debug)]
pub struct TickResult {
    pub t: usize,
    pub out_spiking_channel_ids: Vec<usize>,
    pub spiking_nids: Vec<usize>,
    pub synaptic_transmission_count: usize,
    pub state_snapshot: Option<StateSnapshot>,
}

pub struct Instance {
    out_nid_channel_mapping: HashMap<usize, usize>,
    spiking_nid_buffer: Vec<usize>,
    broadcast_tx: Option<Bus<TickContext>>,
    partition_result_rx: MpscReceiver<PartitionGroupResult>,
    num_partitions: usize,
    tick_period: usize,
    join_handles: Vec<JoinHandle<()>>,
}

impl Instance {
    pub fn tick(
        &mut self,
        in_spike_channel_ids: &[usize],
        reward: f32,
        extract_state_snapshot: bool,
    ) -> TickResult {
        let spike_trigger_nids: Vec<usize> = in_spike_channel_ids.to_vec();

        let spiked_nids: Vec<_> = self.spiking_nid_buffer.drain(0..).collect();

        let dopamine_amount = reward; // to be revised. There might be an indirection via dopaminergic neurons

        let t = self.tick_period;
        let ctx = TickContext {
            t,
            spike_trigger_nids,
            spiked_nids,
            dopamine_amount,
            extract_state_snapshot,
        };

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

        let state_snapshot = if extract_state_snapshot {
            Some(aggregate_state_snapshot(&partition_group_results))
        } else {
            None
        };

        self.spiking_nid_buffer.sort();

        let out_spiking_channel_ids: Vec<usize> = self
            .spiking_nid_buffer
            .iter()
            .filter_map(|spiking_nid| self.out_nid_channel_mapping.get(spiking_nid).copied())
            .collect();

        self.tick_period += 1;

        TickResult {
            out_spiking_channel_ids,
            state_snapshot,
            spiking_nids: self.spiking_nid_buffer.clone(),
            synaptic_transmission_count,
            t,
        }
    }

    pub fn tick_no_input(&mut self) -> TickResult {
        self.tick(&[], 0.0, false)
    }

    pub fn tick_no_input_until(&mut self, t: usize) {
        while self.get_tick_period() < t {
            self.tick_no_input();
        }
    }

    pub fn get_tick_period(&self) -> usize {
        self.tick_period
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

fn aggregate_state_snapshot(partition_group_results: &[PartitionGroupResult]) -> StateSnapshot {
    let neuron_states: Vec<_> = partition_group_results
        .iter()
        .map(|result| result.partition_state_snapshots.as_ref().unwrap())
        .flatten()
        .sorted_by_key(|partition_snapshot| partition_snapshot.nid_start)
        .map(|partition_snapshot| partition_snapshot.neuron_states.iter())
        .flatten()
        .cloned()
        .collect();

    StateSnapshot { neuron_states }
}

#[cfg(test)]
mod tests {
    use crate::params::{LayerParams, NeuronParams};
    use crate::partition::{PartitionGroupResult, PartitionStateSnapshot};
    use crate::state_snapshot::NeuronState;
    use crate::types::HashMap;

    use super::*;
    use float_cmp::assert_approx_eq;
    use std::thread;

    #[test]
    fn simple_round_trip() {
        let num_partitions = 2;

        let broadcast_tx = Some(Bus::new(1));
        let (partition_result_tx, partition_result_rx) = mpsc_channel();

        let spiking_in_channel_id = 3;
        let channel_mapped_nid = 3;
        let spiking_nid = 10;
        let spiking_out_channel_id = 11;

        let out_nid_channel_mapping = HashMap::from_iter([(spiking_nid, spiking_out_channel_id)]);

        let mut instance = Instance {
            out_nid_channel_mapping,
            spiking_nid_buffer: Vec::new(),
            broadcast_tx,
            partition_result_rx,
            num_partitions,
            tick_period: 0,
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
                            partition_state_snapshots: None,
                        })
                        .unwrap();

                    //second cycle
                    let ctx_2nd_cycle = broadcast_rx.recv().unwrap();

                    partition_result_tx
                        .send(PartitionGroupResult {
                            spiking_nids: Vec::new(),
                            synaptic_transmission_count: 0,
                            partition_state_snapshots: None,
                        })
                        .unwrap();

                    (ctx_1st_cycle, ctx_2nd_cycle)
                })
            })
            .collect();

        // first cycle
        let spiking_in_channel_ids = [spiking_in_channel_id];
        let tick_result = instance.tick(&spiking_in_channel_ids, 0.0, false);
        assert_eq!(
            tick_result.out_spiking_channel_ids,
            [spiking_out_channel_id]
        );
        assert_eq!(instance.spiking_nid_buffer, [spiking_nid]);
        assert_eq!(instance.get_tick_period(), 1);

        // second cycle
        let tick_result = instance.tick_no_input();
        assert!(tick_result.out_spiking_channel_ids.is_empty());
        assert!(instance.spiking_nid_buffer.is_empty());
        assert_eq!(instance.get_tick_period(), 2);

        for join_handle in join_handles {
            let (ctx_1st, ctx_2nd) = join_handle.join().unwrap();
            assert_eq!(ctx_1st.spike_trigger_nids, [channel_mapped_nid]);
            assert_eq!(ctx_1st.spiked_nids, [] as [usize; 0]);
            assert_eq!(ctx_2nd.spike_trigger_nids, [] as [usize; 0]);
            assert_eq!(ctx_2nd.spiked_nids, [spiking_nid]);
        }
    }

    #[test]
    fn echo_instance() {
        let num_partitions = 1;

        let broadcast_tx = Some(Bus::new(1));
        let (partition_result_tx, partition_result_rx) = mpsc_channel();
        let mut out_nid_channel_mapping = HashMap::default();

        for i in 0..10 {
            out_nid_channel_mapping.insert(i + 10, i);
        }

        let mut instance = Instance {
            out_nid_channel_mapping,
            spiking_nid_buffer: Vec::new(),
            broadcast_tx,
            partition_result_rx,
            num_partitions,
            tick_period: 0,
            join_handles: Vec::new(),
        };

        let mut broadcast_rx = instance.broadcast_tx.as_mut().unwrap().add_rx();

        let join_handle = thread::spawn(move || {
            while let Ok(ctx) = broadcast_rx.recv() {
                let spiking_nids = ctx
                    .spike_trigger_nids
                    .into_iter()
                    .map(|id| id + 10)
                    .collect();
                partition_result_tx
                    .send(PartitionGroupResult {
                        spiking_nids,
                        synaptic_transmission_count: 0,
                        partition_state_snapshots: None,
                    })
                    .unwrap();
            }
        });

        let inputs = [vec![1, 4, 5], vec![1, 3], vec![], vec![9]];

        for input in inputs {
            let tick_result = instance.tick(&input, 0.0, false);
            assert_eq!(tick_result.out_spiking_channel_ids, input);
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
            neuron_states: vec![NeuronState { voltage: 4.0 }, NeuronState { voltage: 5.0 }],
        };

        let partition_snapshot_1 = PartitionStateSnapshot {
            nid_start: 1,
            neuron_states: vec![
                NeuronState { voltage: 1.0 },
                NeuronState { voltage: 2.0 },
                NeuronState { voltage: 3.0 },
            ],
        };

        let partition_snapshot_2 = PartitionStateSnapshot {
            nid_start: 0,
            neuron_states: vec![NeuronState { voltage: 0.0 }],
        };

        let group_result_0 = PartitionGroupResult {
            spiking_nids: Vec::new(),
            synaptic_transmission_count: 0,
            partition_state_snapshots: Some(vec![partition_snapshot_0, partition_snapshot_2]),
        };

        let group_result_1 = PartitionGroupResult {
            spiking_nids: Vec::new(),
            synaptic_transmission_count: 0,
            partition_state_snapshots: Some(vec![partition_snapshot_1]),
        };

        let group_results = [group_result_0, group_result_1];

        let state_snapshot = aggregate_state_snapshot(&group_results);

        for (index, snapshot) in state_snapshot.neuron_states.iter().enumerate() {
            assert_approx_eq!(f32, snapshot.voltage, index as f32 * 1.0);
        }
    }
}
