use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub neuron_states: Vec<NeuronState>,
    pub synapse_states: Vec<SynapseState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub voltage: f32,
    pub threshold: f32,
    pub is_refractory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseState {
    pub projection_id: usize,
    pub pre_syn_nid: usize,
    pub post_syn_nid: usize,
    pub conduction_delay: u8,
    pub weight: f32,
    pub short_term_stdp_offset: f32,
}
