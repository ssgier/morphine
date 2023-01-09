use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub neuron_states: Vec<NeuronState>,
    pub synapse_states: Vec<SynapseState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub voltage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseState {
    pub pre_syn_nid: usize,
    pub post_syn_nid: usize,
    pub conduction_delay: u8,
    pub weight: f32,
}
