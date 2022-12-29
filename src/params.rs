use serde::{Deserialize, Serialize};
use simple_error::SimpleError;

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
    pub factor_pre_before_post: f32,
    pub tau_pre_before_post: f32,
    pub factor_pre_after_post: f32,
    pub tau_pre_after_post: f32,
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

impl Default for TechnicalParams {
    fn default() -> Self {
        Self {
            num_threads: Some(1),
            pin_threads: false,
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
            factor_pre_before_post: 0.1,
            tau_pre_before_post: 20.0,
            factor_pre_after_post: -0.12,
            tau_pre_after_post: 20.0,
        }
    }
}

pub fn validate_instance_params(instance_params: &InstanceParams) -> Result<(), SimpleError> {
    for layer_params in &instance_params.layers {
        validate_layer_params(layer_params)?;
    }

    for conn_params in &instance_params.layer_connections {
        if conn_params.from_layer_id >= instance_params.layers.len() {
            return Err(SimpleError::new("from_layer_id out of bounds"));
        }

        if conn_params.to_layer_id >= instance_params.layers.len() {
            return Err(SimpleError::new("to_layer_id out of bounds"));
        }

        validate_connection_params(conn_params)?;
    }

    validate_technical_params(&instance_params.technical_params)?;

    Ok(())
}

fn validate_layer_params(layer_params: &LayerParams) -> Result<(), SimpleError> {
    validate_neuron_params(&layer_params.neuron_params)?;

    if let Some(plasticity_modulation_params) = &layer_params.plasticity_modulation_params {
        validate_plasticity_modulation_params(&plasticity_modulation_params)?;
    }

    Ok(())
}

fn validate_connection_params(
    connection_params: &LayerConnectionParams,
) -> Result<(), SimpleError> {
    validate_projection_params(&connection_params.projection_params)?;

    if connection_params.connect_density <= 0.0 {
        return Err(SimpleError::new(
            "connect_density must be strictly positive",
        ));
    }

    if connection_params.connect_width <= 0.0 {
        return Err(SimpleError::new("connect_width must be strictly positive"));
    }

    match connection_params.initial_syn_weight {
        InitialSynWeight::Randomized(max_weight) => {
            if max_weight < 0.0 {
                return Err(SimpleError::new(
                    "Parameter for randomized initial synaptic weight must be strictly positive",
                ));
            }
        }
        InitialSynWeight::Constant(weight) => {
            if weight < 0.0 {
                return Err(SimpleError::new(
                    "Parameter for constant initial synaptic weight must not be negative",
                ));
            }
        }
    }

    if connection_params.conduction_delay_position_distance_scale_factor < 0.0 {
        return Err(SimpleError::new(
            "conduction_delay_position_distance_scale_factor must not be negative",
        ));
    }

    Ok(())
}

fn validate_technical_params(technical_parms: &TechnicalParams) -> Result<(), SimpleError> {
    if let Some(num_threads) = technical_parms.num_threads {
        if num_cpus::get() < num_threads {
            return Err(SimpleError::new(
                "num_threads must not be greater than number of available CPUs",
            ));
        }
    }

    Ok(())
}

fn validate_neuron_params(neuron_params: &NeuronParams) -> Result<(), SimpleError> {
    if neuron_params.tau_membrane <= 0.0 {
        return Err(SimpleError::new("tau_membrane must be strictly positive"));
    }

    if neuron_params.refractory_period == 0 {
        return Err(SimpleError::new(
            "refractory_period must be strictly positive",
        ));
    }

    if neuron_params.reset_voltage >= 1.0 {
        return Err(SimpleError::new("reset_voltage must be less than 1.0"));
    }

    if neuron_params.reset_voltage < neuron_params.voltage_floor {
        return Err(SimpleError::new(
            "reset_voltage must not be less than voltage_floor",
        ));
    }

    if neuron_params.reset_voltage >= neuron_params.adaptation_threshold {
        return Err(SimpleError::new(
            "reset_voltage must be less than adaptation_threshold",
        ));
    }

    if neuron_params.tau_threshold <= 0.0 {
        return Err(SimpleError::new("tau_threshold must be strictly positive"));
    }

    if neuron_params.voltage_floor > 0.0 {
        return Err(SimpleError::new(
            "voltage_floor must not be greater than zero",
        ));
    }

    Ok(())
}

fn validate_plasticity_modulation_params(
    plasticity_modulation_params: &PlasticityModulationParams,
) -> Result<(), SimpleError> {
    if plasticity_modulation_params.tau_eligibility_trace <= 0.0 {
        return Err(SimpleError::new(
            "tau_eligibility_trace must be strictly positive",
        ));
    }

    if plasticity_modulation_params.eligibility_trace_delay
        > plasticity_modulation_params.t_cutoff_eligibility_trace
    {
        return Err(SimpleError::new(
            "eligibility_trace_delay must not be greater than t_cutoff_eligibility_trace",
        ));
    }

    if plasticity_modulation_params.dopamine_modulation_factor <= 0.0 {
        return Err(SimpleError::new(
            "dopamine_modulation_factor must be strictly positive",
        ));
    }

    if plasticity_modulation_params.dopamine_flush_period
        % plasticity_modulation_params.dopamine_conflation_period
        != 0
    {
        return Err(SimpleError::new(
            "dopamine_flush_period must be a multiple of dopamine_conflation_period",
        ));
    }

    Ok(())
}

fn validate_projection_params(prj_params: &ProjectionParams) -> Result<(), SimpleError> {
    validate_synapse_params(&prj_params.synapse_params)?;
    validate_stp_params(&prj_params.stp_params)?;

    if let Some(long_term_stdp_params) = &prj_params.long_term_stdp_params {
        validate_stdp_params(long_term_stdp_params)?;
    }

    if let Some(short_term_stdp_params) = &prj_params.short_term_stdp_params {
        validate_short_term_stdp_params(short_term_stdp_params)?;
    }

    Ok(())
}

fn validate_synapse_params(synapse_params: &SynapseParams) -> Result<(), SimpleError> {
    if synapse_params.max_weight <= 0.0 {
        return Err(SimpleError::new("max_weight must be strictly positive"));
    }

    Ok(())
}

fn validate_stp_params(stp_params: &StpParams) -> Result<(), SimpleError> {
    match *stp_params {
        StpParams::NoStp => Ok(()),
        StpParams::Depression { tau, p0, factor } => validate_stp_params_inner(tau, p0, factor),
        StpParams::Facilitation { tau, p0, factor } => validate_stp_params_inner(tau, p0, factor),
    }
}

fn validate_stp_params_inner(tau: f32, p0: f32, factor: f32) -> Result<(), SimpleError> {
    if tau <= 0.0 {
        return Err(SimpleError::new(
            "stp_params: tau must be strictly positive",
        ));
    }

    if p0 < 0.0 || p0 > 1.0 {
        return Err(SimpleError::new("stp_params: p0 must be in [0, 1]"));
    }

    if factor <= 0.0 || factor > 1.0 {
        return Err(SimpleError::new("stp_params: factor must be in (0, 1]"));
    }

    Ok(())
}

fn validate_stdp_params(stdp_params: &StdpParams) -> Result<(), SimpleError> {
    if stdp_params.tau_pre_before_post <= 0.0 {
        return Err(SimpleError::new(
            "tau_pre_before_post must be strictly positive",
        ));
    }

    if stdp_params.tau_pre_after_post <= 0.0 {
        return Err(SimpleError::new(
            "tau_pre_after_post must be strictly positive",
        ));
    }

    Ok(())
}

fn validate_short_term_stdp_params(
    short_term_stdp_params: &ShortTermStdpParams,
) -> Result<(), SimpleError> {
    validate_stdp_params(&short_term_stdp_params.stdp_params)?;

    if short_term_stdp_params.tau <= 0.0 {
        return Err(SimpleError::new(
            "short_term_stdp_params: tau must be strictly positive",
        ));
    }

    Ok(())
}
