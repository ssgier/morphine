use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct InstanceParams {
    pub layers: Vec<LayerParams>,
    pub layer_connections: Vec<LayerConnectionParams>,
    pub technical_params: TechnicalParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParams {
    pub num_neurons: usize,
    pub neuron_params: NeuronParams,
    pub plasticity_modulation_params: Option<PlasticityModulationParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConnectionParams {
    pub from_layer_id: usize,
    pub to_layer_id: usize,
    pub projection_params: ProjectionParams,
    pub connect_density: f64,
    pub connect_width: f64,
    pub initial_syn_weight: InitialSynWeight,
    pub conduction_delay_max_random_part: usize,
    pub conduction_delay_position_distance_scale_factor: f64,
    pub conduction_delay_add_on: usize,
}

impl LayerConnectionParams {
    pub fn defaults_for_layer_ids(from_layer_id: usize, to_layer_id: usize) -> Self {
        Self {
            from_layer_id,
            to_layer_id,
            projection_params: ProjectionParams::default(),
            connect_density: 1.0,
            connect_width: 2.0,
            initial_syn_weight: InitialSynWeight::Constant(1.0),
            conduction_delay_max_random_part: 0,
            conduction_delay_position_distance_scale_factor: 0.0,
            conduction_delay_add_on: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectionParams {
    pub synapse_params: SynapseParams,
    pub stp_params: StpParams,
    pub long_term_stdp_params: Option<StdpParams>,
    pub short_term_stdp_params: Option<ShortTermStdpParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitialSynWeight {
    Randomized(f32),
    Constant(f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseParams {
    pub max_weight: f32,
    pub weight_scale_factor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StpParams {
    NoStp,
    Depression { tau: f32, p0: f32, factor: f32 },
    Facilitation { tau: f32, p0: f32, factor: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdpParams {
    pub factor_potentiation: f32,
    pub tau_potentiation: f32,
    pub factor_depression: f32,
    pub tau_depression: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortTermStdpParams {
    pub stdp_params: StdpParams,
    pub tau: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityModulationParams {
    pub tau_eligibility_trace: f32,
    pub eligibility_trace_delay: usize,
    pub dopamine_modulation_factor: f32,
    pub t_cutoff_eligibility_trace: usize,
    pub dopamine_flush_period: usize,
    pub dopamine_conflation_period: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopamineBufferingParams {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronParams {
    pub tau_membrane: f32,
    pub refractory_period: u8,
    pub reset_voltage: f32,
    pub t_cutoff_coincidence: usize,
    pub adaptation_threshold: f32,
    pub tau_threshold: f32,
    pub voltage_floor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalParams {
    pub num_threads: Option<usize>,
    pub pin_threads: bool,
    pub batched_ring_buffer_size: usize,
}

impl Default for LayerParams {
    fn default() -> Self {
        Self {
            num_neurons: 1,
            neuron_params: NeuronParams::default(),
            plasticity_modulation_params: None,
        }
    }
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            tau_membrane: 10.0,
            refractory_period: 10,
            reset_voltage: 0.0,
            t_cutoff_coincidence: 20,
            adaptation_threshold: 1.0,
            tau_threshold: 50.0,
            voltage_floor: 0.0,
        }
    }
}

const DEFAULT_MAX_CONDUCTION_DELAY: usize = 20;

impl Default for TechnicalParams {
    fn default() -> Self {
        Self {
            num_threads: Some(1),
            pin_threads: false,
            batched_ring_buffer_size: DEFAULT_MAX_CONDUCTION_DELAY + 1,
        }
    }
}

impl Default for SynapseParams {
    fn default() -> Self {
        Self {
            max_weight: 1.0,
            weight_scale_factor: 1.0,
        }
    }
}

impl Default for PlasticityModulationParams {
    fn default() -> Self {
        Self {
            tau_eligibility_trace: 500.0,
            eligibility_trace_delay: 0,
            t_cutoff_eligibility_trace: 1000,
            dopamine_flush_period: 1000,
            dopamine_conflation_period: 200,
            dopamine_modulation_factor: 1.0,
        }
    }
}

impl Default for StpParams {
    fn default() -> Self {
        StpParams::NoStp
    }
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            factor_potentiation: 0.1,
            tau_potentiation: 20.0,
            factor_depression: -0.12,
            tau_depression: 20.0,
        }
    }
}
