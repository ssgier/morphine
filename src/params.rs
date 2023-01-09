use serde::{Deserialize, Serialize};
use simple_error::SimpleError;

use crate::types::HashSet;

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
    pub allow_self_innervation: bool,
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
            allow_self_innervation: true,
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
    pub seed_override: Option<u64>,
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
            seed_override: None,
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

    let mut seen_from_to_pairs = HashSet::default();

    for conn_params in &instance_params.layer_connections {
        if !seen_from_to_pairs.insert((conn_params.from_layer_id, conn_params.to_layer_id)) {
            return Err(SimpleError::new(format!(
                "duplicate connection from layer {} to layer {}",
                conn_params.from_layer_id, conn_params.to_layer_id
            )));
        }

        if conn_params.from_layer_id >= instance_params.layers.len() {
            return Err(SimpleError::new(format!(
                "invalid from_layer_id: {}",
                conn_params.from_layer_id
            )));
        }

        if conn_params.to_layer_id >= instance_params.layers.len() {
            return Err(SimpleError::new(format!(
                "invalid to_layer_id: {}",
                conn_params.to_layer_id
            )));
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

    if connection_params.connect_density <= 0.0 || connection_params.connect_density > 1.0 {
        return Err(SimpleError::new("connect_density must be in (0, 1]"));
    }

    if connection_params.connect_width <= 0.0 || connection_params.connect_width > 2.0 {
        return Err(SimpleError::new("connect_width must be in (0, 2]"));
    }

    match connection_params.initial_syn_weight {
        InitialSynWeight::Randomized(max_weight) => {
            if max_weight <= 0.0 {
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::util::test_util;

    #[test]
    fn valid_params() {
        let params = test_util::get_template_instance_params();
        assert!(validate_instance_params(&params).is_ok());
    }

    #[test]
    fn zero_tau_membrane() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.tau_membrane = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "tau_membrane must be strictly positive"
        );
    }

    #[test]
    fn zero_refractory_period() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.refractory_period = 0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "refractory_period must be strictly positive"
        );
    }

    #[test]
    fn too_high_reset_voltage() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.reset_voltage = 1.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "reset_voltage must be less than 1.0"
        );
    }

    #[test]
    fn reset_voltage_lower_than_floor() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.reset_voltage = -0.1;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "reset_voltage must not be less than voltage_floor"
        );
    }

    #[test]
    fn reset_voltage_not_less_than_adaptation_threshold() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.adaptation_threshold = 0.5;
        params.layers[0].neuron_params.reset_voltage = 0.5;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "reset_voltage must be less than adaptation_threshold"
        );
    }

    #[test]
    fn zero_tau_threshold() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.tau_threshold = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "tau_threshold must be strictly positive"
        );
    }

    #[test]
    fn voltage_floor_greater_than_zero() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0].neuron_params.reset_voltage = 0.1;
        params.layers[0].neuron_params.voltage_floor = 0.1;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "voltage_floor must not be greater than zero"
        );
    }

    #[test]
    fn zero_tau_elig_trace() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0]
            .plasticity_modulation_params
            .as_mut()
            .unwrap()
            .tau_eligibility_trace = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "tau_eligibility_trace must be strictly positive"
        );
    }

    #[test]
    fn elig_trace_delay_greater_than_cutoff() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0]
            .plasticity_modulation_params
            .as_mut()
            .unwrap()
            .eligibility_trace_delay = 1501;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "eligibility_trace_delay must not be greater than t_cutoff_eligibility_trace"
        );
    }

    #[test]
    fn zero_dopamine_modulation_factor() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0]
            .plasticity_modulation_params
            .as_mut()
            .unwrap()
            .dopamine_modulation_factor = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "dopamine_modulation_factor must be strictly positive"
        );
    }

    #[test]
    fn flush_period_nod_multiple_of_conflation_period() {
        let mut params = test_util::get_template_instance_params();
        params.layers[0]
            .plasticity_modulation_params
            .as_mut()
            .unwrap()
            .dopamine_flush_period = 251;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "dopamine_flush_period must be a multiple of dopamine_conflation_period"
        );
    }

    #[test]
    fn duplicate_connection() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].from_layer_id = 1;
        params.layer_connections[0].to_layer_id = 0;
        params.layer_connections[1].from_layer_id = 1;
        params.layer_connections[1].to_layer_id = 0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "duplicate connection from layer 1 to layer 0"
        );
    }

    #[test]
    fn invalid_from_layer_id() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].from_layer_id = 2;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(result.unwrap_err().as_str(), "invalid from_layer_id: 2");
    }

    #[test]
    fn invalid_to_layer_id() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].to_layer_id = 2;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(result.unwrap_err().as_str(), "invalid to_layer_id: 2");
    }

    #[test]
    fn zero_connect_density() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].connect_density = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "connect_density must be in (0, 1]"
        );
    }

    #[test]
    fn too_high_connect_density() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].connect_density = 1.1;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "connect_density must be in (0, 1]"
        );
    }

    #[test]
    fn zero_connect_width() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].connect_width = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "connect_width must be in (0, 2]"
        );
    }

    #[test]
    fn too_high_connect_width() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].connect_width = 2.1;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "connect_width must be in (0, 2]"
        );
    }

    #[test]
    fn zero_initial_weight_randomized() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].initial_syn_weight = InitialSynWeight::Randomized(0.0);
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "Parameter for randomized initial synaptic weight must be strictly positive"
        );
    }

    #[test]
    fn negative_initial_weight_constant() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].initial_syn_weight = InitialSynWeight::Constant(-0.1);
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "Parameter for constant initial synaptic weight must not be negative"
        );
    }

    #[test]
    fn negative_conduction_delay_distance_scale_factor() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].conduction_delay_position_distance_scale_factor = -0.1;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "conduction_delay_position_distance_scale_factor must not be negative"
        );
    }

    #[test]
    fn zero_max_weight() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0]
            .projection_params
            .synapse_params
            .max_weight = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "max_weight must be strictly positive"
        );
    }

    #[test]
    fn zero_tau_stp_facilitation() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].projection_params.stp_params = StpParams::Facilitation {
            tau: 0.0,
            p0: 0.5,
            factor: 0.5,
        };
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "stp_params: tau must be strictly positive"
        );
    }

    #[test]
    fn negative_p0_facilitation() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].projection_params.stp_params = StpParams::Facilitation {
            tau: 100.0,
            p0: -0.1,
            factor: 0.5,
        };
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "stp_params: p0 must be in [0, 1]"
        );
    }

    #[test]
    fn zero_factor_facilitation() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].projection_params.stp_params = StpParams::Facilitation {
            tau: 100.0,
            p0: 0.5,
            factor: 0.0,
        };
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "stp_params: factor must be in (0, 1]"
        );
    }

    #[test]
    fn zero_tau_stp_depression() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].projection_params.stp_params = StpParams::Depression {
            tau: 0.0,
            p0: 0.5,
            factor: 0.5,
        };
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "stp_params: tau must be strictly positive"
        );
    }

    #[test]
    fn too_high_p0_depression() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].projection_params.stp_params = StpParams::Depression {
            tau: 100.0,
            p0: 1.1,
            factor: 0.5,
        };
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "stp_params: p0 must be in [0, 1]"
        );
    }

    #[test]
    fn too_high_factor_depression() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0].projection_params.stp_params = StpParams::Depression {
            tau: 100.0,
            p0: 0.5,
            factor: 1.1,
        };
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "stp_params: factor must be in (0, 1]"
        );
    }

    #[test]
    fn zero_tau_pre_before_post_short_term_stdp() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0]
            .projection_params
            .short_term_stdp_params
            .as_mut()
            .unwrap()
            .stdp_params
            .tau_pre_before_post = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "tau_pre_before_post must be strictly positive"
        );
    }

    #[test]
    fn zero_tau_pre_after_post_long_term_stdp() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0]
            .projection_params
            .long_term_stdp_params
            .as_mut()
            .unwrap()
            .tau_pre_after_post = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "tau_pre_after_post must be strictly positive"
        );
    }

    #[test]
    fn zero_tau_short_term_stdp() {
        let mut params = test_util::get_template_instance_params();
        params.layer_connections[0]
            .projection_params
            .short_term_stdp_params
            .as_mut()
            .unwrap()
            .tau = 0.0;
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "short_term_stdp_params: tau must be strictly positive"
        );
    }

    #[test]
    fn too_high_num_threads() {
        let mut params = test_util::get_template_instance_params();
        params.technical_params.num_threads = Some(num_cpus::get() + 1);
        let result = validate_instance_params(&params);

        assert!(result.is_err());

        assert_eq!(
            result.unwrap_err().as_str(),
            "num_threads must not be greater than number of available CPUs"
        );
    }
}
