use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub neuron_states: Vec<NeuronState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub voltage: f32,
}
