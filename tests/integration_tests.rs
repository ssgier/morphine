use std::collections::HashSet;
use std::{collections::hash_map::DefaultHasher, hash::Hasher};

use float_cmp::assert_approx_eq;
use itertools::{assert_equal, Itertools};
use morphine::{
    api::SCMode,
    instance::{self, create_instance, Instance, TickInput, TickResult},
    params::{
        InitialSynWeight, InstanceParams, LayerConnectionParams, LayerParams,
        PlasticityModulationParams, ShortTermStdpParams, StdpParams, StpParams,
    },
};
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng,
};

fn tick(instance: &mut Instance, in_channel_ids: &[usize]) -> TickResult {
    instance
        .tick(&TickInput::from_spiking_in_channel_ids(in_channel_ids))
        .unwrap()
}

fn make_simple_1_in_1_out_instance(weight: f32) -> Instance {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.tau_membrane = 10.0;
    layer.neuron_params.refractory_period = 10;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(weight);
    params.layer_connections.push(connection_params);

    create_instance(params).unwrap()
}

const STDP_PARAMS: StdpParams = StdpParams {
    factor_pre_before_post: 0.1,
    tau_pre_before_post: 20.0,
    factor_pre_after_post: -0.11,
    tau_pre_after_post: 25.0,
};

#[test]
fn empty_instance() {
    let params = InstanceParams::default();
    let mut instance = create_instance(params).unwrap();
    let tick_0_result = instance.tick_no_input();
    assert!(tick_0_result.spiking_out_channel_ids.is_empty());
    assert!(tick_0_result.spiking_nids.is_empty());
}

#[test]
fn empty_output_layer() {
    let mut params = InstanceParams::default();
    let mut layer_params = LayerParams::default();
    layer_params.num_neurons = 1;
    params.layers.push(layer_params.clone());
    layer_params.num_neurons = 0;
    params.layers.push(layer_params);
    let mut instance = create_instance(params).unwrap();
    let tick_0_result = tick(&mut instance, &[0]);
    assert!(tick_0_result.spiking_out_channel_ids.is_empty());
    assert_equal(tick_0_result.spiking_nids, [0]);
}

#[test]
fn single_neuron() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 1;

    params.layers.push(layer);

    let mut instance = create_instance(params).unwrap();

    // t = 0: empty context, expect empty result
    let tick_0_result = tick(&mut instance, &[]);
    assert!(tick_0_result.spiking_out_channel_ids.is_empty());

    // t = 1: input channel 0 ticks, expect output channel 0 tick
    let tick_1_result = tick(&mut instance, &[0]);
    assert_equal(tick_1_result.spiking_out_channel_ids, [0]);
}

#[test]
fn single_direct_mapped_output() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    // double spike on nid 0 ...
    let tick_0_result = tick(&mut instance, &[0, 0]);
    assert!(tick_0_result.spiking_out_channel_ids.is_empty());

    // ... leads to spike at nid 1 at next tick, mapped to output channel 0
    let tick_1_result = instance.tick_no_input();
    assert_equal(tick_1_result.spiking_out_channel_ids, [0]);
}

#[test]
fn missed_spike_after_leakage() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    // first spike on nid 0
    let tick_0_result = tick(&mut instance, &[0]);
    assert!(tick_0_result.spiking_out_channel_ids.is_empty());

    // seconds spike on nid 0
    let tick_1_result = tick(&mut instance, &[0]);
    assert!(tick_1_result.spiking_out_channel_ids.is_empty());

    // spike is missed, but voltaged is close to threshold
    let tick_2_result = tick(&mut instance, &[]);
    assert!(tick_2_result.spiking_out_channel_ids.is_empty());

    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        0.5 * (-1.0 / 10.0f32).exp() + 0.5
    );
}

#[test]
fn voltage_trajectory() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    tick(&mut instance, &[0]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.5);

    instance.tick_no_input_until_inclusive(7);

    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        0.5 * (-5.0 / 10.0f32).exp()
    );
}

#[test]
fn no_psp_during_refractory_period() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    // make neuron 1 fire one tick later
    tick(&mut instance, &[0, 0]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.0);

    tick(&mut instance, &[0]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    // neuron 1 is in refractory period
    assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.0);

    while instance.get_last_tick_period() < 9 {
        instance.tick_no_input();
    }

    // too early to cause a psp
    tick(&mut instance, &[0]);
    tick(&mut instance, &[0]);
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.0);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    // neuron 1 is not in refractory period anymore
    assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, 0.5);
}

#[test]
fn two_epsps_and_one_ipsp() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.tau_membrane = 10.0;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    params.layer_connections.push(connection_params);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 2);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params.conduction_delay_add_on = 1;
    params.layer_connections.push(connection_params);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(1, 2);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_params
        .projection_params
        .synapse_params
        .weight_scale_factor = -1.0; // inhibition

    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    instance.tick_no_input(); // start with an empty tick for test diversity

    // double tick on channel 0
    tick(&mut instance, &[0, 0]);

    let tick_2_result = tick(&mut instance, &[]);

    assert_equal(tick_2_result.spiking_nids, [1]);
    assert_equal(tick_2_result.spiking_out_channel_ids, [] as [usize; 0]);

    let tick_3_result = instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    assert!(tick_3_result.spiking_nids.is_empty());
    assert!(tick_3_result.spiking_out_channel_ids.is_empty());
    assert_approx_eq!(f32, state_snapshot.neuron_states[2].voltage, 0.9);
}

#[test]
fn voltage_floor() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.tau_membrane = 10.0;
    layer.neuron_params.refractory_period = 10;
    layer.neuron_params.voltage_floor = -0.6;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params
        .projection_params
        .synapse_params
        .weight_scale_factor = -1.0;

    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    // double tick on channel 0
    tick(&mut instance, &[0, 0]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    // voltage floored at -0.6
    assert_approx_eq!(f32, state_snapshot.neuron_states[1].voltage, -0.6);
}

#[test]
fn threshold_adaptation() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 1;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    layer.neuron_params.adaptation_threshold = 0.4;
    layer.neuron_params.tau_threshold = 10.0;
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    // trigger spike in nid 1
    tick(&mut instance, &[0, 0]);

    // nid 1 spikes, threshold drops to 0.4
    instance.tick_no_input();

    // now a psp of 0.5 is sufficient to spike
    tick(&mut instance, &[0]);
    let tick_3_result = instance.tick_no_input();
    assert_equal(tick_3_result.spiking_out_channel_ids, [0]);

    instance.tick_no_input();
    instance.tick_no_input();

    // threshold offset decayed and psp of 0.5 is not sufficient anymore
    tick(&mut instance, &[0]);
    let tick_7_result = instance.tick_no_input();
    assert!(tick_7_result.spiking_out_channel_ids.is_empty());
}

#[test]
fn simple_potentiation_long_term_stdp() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 1;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    layer.neuron_params.adaptation_threshold = 2.0;
    layer.neuron_params.tau_threshold = 10.0;
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(1.0);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);
    instance.tick_no_input(); // nid 1 spikes

    tick(&mut instance, &[0]);
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.synapse_states[0].weight, 1.1);

    let tick_3_result = instance.tick_no_input();
    assert!(tick_3_result.spiking_out_channel_ids.is_empty());

    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        1.1 // increased psp after potentiation
    );
}

#[test]
fn synaptic_transmission_count() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.tau_membrane = 10.0;
    layer.neuron_params.refractory_period = 10;
    layer.neuron_params.t_cutoff_coincidence = 10;
    layer.num_neurons = 10;
    params.layers.push(layer.clone());
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params.conduction_delay_position_distance_scale_factor = 0.0;
    connection_params.connect_width = 2.0;
    connection_params.connect_density = 1.0;
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.0;
    params.layer_connections.push(connection_params.clone());
    connection_params.to_layer_id = 1;
    connection_params.connect_density = 0.5;
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);
    let tick_1_result = instance.tick_no_input();

    assert_eq!(tick_1_result.synaptic_transmission_count, 15);
}

#[test]
fn simple_potentiation_short_term_stdp() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 1;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    layer.neuron_params.adaptation_threshold = 2.0;
    layer.neuron_params.tau_threshold = 10.0;
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(1.0);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.projection_params.short_term_stdp_params = Some(ShortTermStdpParams {
        stdp_params: STDP_PARAMS,
        tau: 1.0,
    });
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);
    instance.tick_no_input(); // nid 1 spikes
    instance.tick_no_input();

    tick(&mut instance, &[0]);

    let tick_3_result = instance.tick_no_input();
    assert!(tick_3_result.spiking_out_channel_ids.is_empty());

    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        1.0 + 0.1 * (-3.0f32).exp()
    );

    assert_approx_eq!(
        f32,
        state_snapshot.synapse_states[0].weight,
        1.0 // unchanged long term weight
    );
}

#[test]
fn pre_syn_spike_then_two_post_syn_spikes() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 10;
    layer.num_neurons = 2;
    layer.neuron_params.tau_threshold = 10.0;
    layer.neuron_params.t_cutoff_coincidence = 10;
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.4);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.0;
    connection_params.connect_width = 2.0;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);

    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);

    tick(&mut instance, &[1]);
    instance.tick_no_input_until_inclusive(3);
    tick(&mut instance, &[1]);

    instance.tick_no_input_until_inclusive(20);
    tick(&mut instance, &[0]);
    let state_snapshot = instance.extract_state_snapshot();

    assert_eq!(state_snapshot.synapse_states[1].pre_syn_nid, 0);
    assert_eq!(state_snapshot.synapse_states[1].post_syn_nid, 1);

    assert_approx_eq!(
        f32,
        state_snapshot.synapse_states[1].weight,
        0.5 // synapse potentiated, but only once
    );

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        0.5 // synapse potentiated, but only once
    );
}

#[test]
fn post_syn_spike_then_two_pre_syn_spikes() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 10;
    layer.num_neurons = 2;
    layer.neuron_params.tau_threshold = 10.0;
    layer.neuron_params.t_cutoff_coincidence = 10;
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.4);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.connect_width = 2.0;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);

    connection_params.projection_params.short_term_stdp_params = Some(ShortTermStdpParams {
        stdp_params: StdpParams {
            factor_pre_before_post: 0.05,
            tau_pre_before_post: 15.0,
            factor_pre_after_post: -0.06,
            tau_pre_after_post: 20.0,
        },
        tau: 10.0,
    });

    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);

    for _tick_period in 1..5 {
        instance.tick_no_input();
    }

    tick(&mut instance, &[1]);
    instance.tick_no_input(); // psp arrives at tick 6, contributes to stdp
    tick(&mut instance, &[1]);
    instance.tick_no_input(); // psp arrives at tick 8, does not contribute to stdp
    tick(&mut instance, &[1]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    let expected_lt_stdp_value_tick_6 = -0.11 * (-6.0 / 25.0f32).exp();
    let expected_st_stdp_value_tick_6 = -0.06 * (-6.0 / 20.0f32).exp();

    let expected_weight = 0.4 + expected_lt_stdp_value_tick_6;
    let expected_st_stdp_offset_tick_10 = expected_st_stdp_value_tick_6 * (-4.0 / 10.0f32).exp();

    let expected_psp = expected_weight + expected_st_stdp_offset_tick_10;

    assert_approx_eq!(f32, state_snapshot.neuron_states[0].voltage, expected_psp);
}

#[test]
fn stdp_alternating_pre_post_syn_spikes() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 15;
    layer.num_neurons = 2;
    layer.neuron_params.t_cutoff_coincidence = 20;
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.4);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.conduction_delay_add_on = 2;
    connection_params.connect_width = 2.0;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);

    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[1]);
    instance.tick_no_input_until_inclusive(8);
    tick(&mut instance, &[0]); // psp at t = 11
    instance.tick_no_input_until_inclusive(12);
    tick(&mut instance, &[1]);
    instance.tick_no_input_until_inclusive(21);
    tick(&mut instance, &[0]); // psp at t = 24
    instance.tick_no_input_until_inclusive(30);
    tick(&mut instance, &[0]); // psp at t = 33
    instance.tick_no_input_until_inclusive(33);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    let tick_11_stdp = -0.11 * (-11.0 / 25.0f32).exp();
    let tick_12_stdp = 0.1 * (-1.0 / 20.0f32).exp();
    let tick_24_stdp = -0.11 * (-12.0 / 25.0f32).exp();
    let expected_weight = 0.4 + tick_11_stdp + tick_12_stdp + tick_24_stdp;

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        expected_weight
    );
}

#[test]
fn long_term_stdp_complex_scenario() {
    // nid 0 and 9 fire such that their spikes arrive at nid 19 simultaneously, triggering a spike.
    // nid 5 also fires before nid 19, but the psp arrives after nid 19 has spiked.

    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.tau_membrane = 10.0;
    layer.neuron_params.refractory_period = 10;
    layer.neuron_params.t_cutoff_coincidence = 10;
    layer.num_neurons = 10;
    params.layers.push(layer.clone());
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params.conduction_delay_position_distance_scale_factor = 10.0;
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.0;
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);
    instance.tick_no_input_until_inclusive(8);

    //this spike will arrive at t = 13, after the post-synaptic spike -> depression
    tick(&mut instance, &[5]);

    instance.tick_no_input_until_inclusive(10);
    tick(&mut instance, &[9]);

    // spikes from neuron 0 and 9 should arrive at neuron 19 (output channel 9) simultaneously
    let tick_11_result = instance.tick_no_input();
    assert_equal(tick_11_result.spiking_out_channel_ids, [9]);

    instance.tick_no_input_until_inclusive(25);

    tick(&mut instance, &[9]);

    tick(&mut instance, &[9]);

    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[19].voltage,
        0.6 // synapse potentiated
    );

    // make nid 19 spike to bring it back to reset voltage
    let tick_27_result = tick(&mut instance, &[9]);
    assert_equal(tick_27_result.spiking_out_channel_ids, [9]);

    // this will cause a psp of psp due to potentiation
    tick(&mut instance, &[0]);

    // cause as spike at nid 19 later on to bring it back to reset voltage
    tick(&mut instance, &[0]);

    instance.tick_no_input_until_inclusive(39);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(f32, state_snapshot.neuron_states[19].voltage, 0.6);

    let tick_40_result = instance.tick_no_input();
    assert_equal(tick_40_result.spiking_out_channel_ids, [9]);

    instance.tick_no_input_until_inclusive(50);
    tick(&mut instance, &[5]);
    instance.tick_no_input_until_inclusive(55);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[19].voltage,
        0.5 - 0.11 * (-2.0 / 25.0f32).exp() // synapse depressed
    );

    // again: induce spike to get back to reset voltage
    tick(&mut instance, &[9, 9]);
    let tick_57_result = instance.tick_no_input();
    assert_equal(tick_57_result.spiking_out_channel_ids, [9]);

    instance.tick_no_input_until_inclusive(70);

    // psp will arrive at t = 74
    tick(&mut instance, &[6]);
    instance.tick_no_input_until_inclusive(74);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[19].voltage,
        0.5 // unaltered synapse
    );
}

#[test]
fn no_dopamine() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 1;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    layer.plasticity_modulation_params = Some(PlasticityModulationParams {
        tau_eligibility_trace: 1000.0,
        eligibility_trace_delay: 0,
        dopamine_modulation_factor: 1.0,
        t_cutoff_eligibility_trace: 2000,
        dopamine_flush_period: 100,
        dopamine_conflation_period: 10,
    });
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 0]);
    instance.tick_no_input(); // nid 1 spikes
    instance.tick_no_input_until_inclusive(1500);
    tick(&mut instance, &[0]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        0.5 // unchanged synapse
    );
}

#[test]
fn negative_reward() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    let mut plasticity_modulation_params = PlasticityModulationParams::default();
    plasticity_modulation_params.dopamine_conflation_period = 1;
    plasticity_modulation_params.dopamine_flush_period = 1;
    plasticity_modulation_params.tau_eligibility_trace = 15.0;
    layer.plasticity_modulation_params = Some(plasticity_modulation_params);
    params.layers.push(layer);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.initial_syn_weight = InitialSynWeight::Constant(1.0);
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);
    tick(&mut instance, &[]);

    let tick_input = TickInput::from_reward(-1.0);
    instance.tick(&tick_input).unwrap();

    let expected_depression = 0.1 * (-1.0 / 15.0f32).exp();

    let synapse_states = instance.extract_state_snapshot().synapse_states;
    assert_approx_eq!(f32, synapse_states[0].weight, 1.0 - expected_depression);
}

#[test]
fn simple_dopamine_scenario() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 1;
    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    layer.plasticity_modulation_params = Some(PlasticityModulationParams {
        tau_eligibility_trace: 1000.0,
        eligibility_trace_delay: 0,
        dopamine_modulation_factor: 0.3,
        t_cutoff_eligibility_trace: 2000,
        dopamine_flush_period: 100,
        dopamine_conflation_period: 10,
    });
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    // two tick at nid 0 cause tick at nid 1 next cycle, creating an eligibility trace
    tick(&mut instance, &[0, 0]);

    instance.tick_no_input_until_inclusive(53);

    // dopamine released at t = 60
    instance.tick(&TickInput::from_reward(1.5)).unwrap();

    instance.tick_no_input_until_inclusive(1500);

    tick(&mut instance, &[0]);

    let state_snapshot = instance.extract_state_snapshot();

    let stdp_value = 0.2; // 2 * 0.1 (two transmissions at same synapse)
    let elig_trace_value = 1.5 * (-58.0 / 1000f32).exp();
    let expected_weight = 0.5 + 0.3 * stdp_value * elig_trace_value;

    assert_approx_eq!(
        f32,
        state_snapshot.synapse_states[0].weight,
        expected_weight
    );

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[1].voltage,
        expected_weight
    );
}

#[test]
fn short_term_plasticity() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 1;
    layer.num_neurons = 2;
    params.layers.push(layer.clone());
    params.layers.push(layer);
    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_params.conduction_delay_position_distance_scale_factor = 5.0;
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 1.5;
    connection_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_params.projection_params.stp_params = StpParams::Depression {
        tau: 800.0,
        p0: 0.8,
        factor: 0.6,
    };
    params.layer_connections.push(connection_params);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0]);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.neuron_states[2].voltage, 0.5 * 0.8);

    instance.tick_no_input_until_inclusive(6);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.neuron_states[3].voltage, 0.5 * 0.8);

    instance.tick_no_input_until_inclusive(500); // let voltages decay to near zero

    tick(&mut instance, &[1]);
    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();
    assert_approx_eq!(f32, state_snapshot.neuron_states[3].voltage, 0.5 * 0.8);

    instance.tick_no_input_until_inclusive(1000);

    // both outgoing synapses of nid 0 are still depressed
    tick(&mut instance, &[0]);
    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    let expected_stp_factor = 0.8 * (1.0 - 0.6 * (-1000.0 / 800.0f32).exp());

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[2].voltage,
        0.5 * expected_stp_factor
    );

    instance.tick_no_input_until_inclusive(1006);

    instance.tick_no_input();
    let state_snapshot = instance.extract_state_snapshot();

    assert_approx_eq!(
        f32,
        state_snapshot.neuron_states[3].voltage,
        0.5 * expected_stp_factor
    );
}

fn get_scenario_template_params() -> InstanceParams {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 10;
    layer.num_neurons = 800;

    layer.plasticity_modulation_params = Some(PlasticityModulationParams {
        tau_eligibility_trace: 1000.0,
        eligibility_trace_delay: 20,
        dopamine_modulation_factor: 1.5,
        t_cutoff_eligibility_trace: 1000,
        dopamine_flush_period: 100,
        dopamine_conflation_period: 50,
    });

    params.layers.push(layer.clone());
    layer.plasticity_modulation_params = None;
    layer.num_neurons = 200;
    layer.neuron_params.tau_membrane = 4.0;
    layer.neuron_params.refractory_period = 5;
    params.layers.push(layer);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    connection_params.initial_syn_weight = InitialSynWeight::Randomized(0.5);
    connection_params.conduction_delay_position_distance_scale_factor = 0.0;
    connection_params.connect_width = 2.0;
    connection_params.connect_density = 0.1;
    connection_params.conduction_delay_max_random_part = 20;
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 0.5;
    connection_params.projection_params.long_term_stdp_params = Some(StdpParams::default());
    connection_params.projection_params.short_term_stdp_params = Some(ShortTermStdpParams {
        stdp_params: StdpParams {
            factor_pre_before_post: 0.01,
            tau_pre_before_post: 20.0,
            factor_pre_after_post: 0.012,
            tau_pre_after_post: 20.0,
        },
        tau: 500.0,
    });
    connection_params.projection_params.stp_params = StpParams::Depression {
        tau: 800.0,
        p0: 0.9,
        factor: 0.2,
    };
    params.layer_connections.push(connection_params.clone());
    connection_params.connect_density = 0.25;
    connection_params.to_layer_id = 1;
    connection_params
        .projection_params
        .synapse_params
        .weight_scale_factor = 2.0;
    params.layer_connections.push(connection_params.clone());

    connection_params.from_layer_id = 1;
    connection_params.to_layer_id = 0;

    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.85);
    connection_params.projection_params.long_term_stdp_params = None;
    connection_params.projection_params.short_term_stdp_params = None;
    connection_params.projection_params.stp_params = StpParams::NoStp;
    connection_params.conduction_delay_max_random_part = 0;
    connection_params
        .projection_params
        .synapse_params
        .weight_scale_factor = -1.0;

    params.layer_connections.push(connection_params.clone());
    connection_params.to_layer_id = 1;
    params.layer_connections.push(connection_params);

    params
}

fn assert_equivalence(
    instances: &mut [Instance],
    t_stop: usize,
    with_reward: bool,
    compare_synapse_states: bool,
    compare_syn_transmission_counts: bool,
) {
    let all_in_channels: Vec<usize> = (0..instances[0].get_num_in_channels()).collect();
    let mut rng = StdRng::seed_from_u64(0);
    let reward_dist = Uniform::new(0.0, 0.05);

    let mut tick_input = TickInput::new();

    for _ in 0..t_stop {
        tick_input.reset();

        tick_input.spiking_in_channel_ids = all_in_channels
            .choose_multiple(&mut rng, 5)
            .copied()
            .collect();

        if with_reward {
            tick_input.reward = reward_dist.sample(&mut rng);
        }

        let mut tick_results = instances
            .iter_mut()
            .map(|instance| instance.tick(&tick_input).unwrap())
            .collect_vec();

        let cmp_result = tick_results.pop().unwrap();

        for tick_result in tick_results {
            assert_eq!(
                tick_result.spiking_out_channel_ids,
                cmp_result.spiking_out_channel_ids
            );
            assert_eq!(tick_result.spiking_nids, cmp_result.spiking_nids);
            if compare_syn_transmission_counts {
                assert_eq!(
                    tick_result.synaptic_transmission_count,
                    cmp_result.synaptic_transmission_count
                );
            }
        }
    }
    tick_input.reset();

    instances
        .iter_mut()
        .map(|instance| instance.tick(&tick_input).unwrap())
        .collect_vec();

    let mut state_snapshots = instances
        .iter_mut()
        .map(|instance| instance.extract_state_snapshot())
        .collect_vec();

    let cmp_state_snapshot = state_snapshots.pop().unwrap();

    for state_snapshot in state_snapshots {
        for (neuron_state, cmp_neuron_state) in state_snapshot
            .neuron_states
            .iter()
            .zip(cmp_state_snapshot.neuron_states.iter())
        {
            assert_approx_eq!(f32, neuron_state.voltage, cmp_neuron_state.voltage);
            assert_approx_eq!(f32, neuron_state.threshold, cmp_neuron_state.threshold);
            assert_eq!(neuron_state.is_refractory, cmp_neuron_state.is_refractory);
        }

        if compare_synapse_states {
            let syn_states = state_snapshot
                .synapse_states
                .iter()
                .sorted_unstable_by_key(|ss| (ss.pre_syn_nid, ss.post_syn_nid))
                .collect_vec();

            let cmp_syn_states = cmp_state_snapshot
                .synapse_states
                .iter()
                .sorted_unstable_by_key(|ss| (ss.pre_syn_nid, ss.post_syn_nid))
                .collect_vec();

            for (syn_state, cmp_syn_state) in syn_states.iter().zip(cmp_syn_states.iter()) {
                assert_eq!(syn_state.pre_syn_nid, cmp_syn_state.pre_syn_nid);
                assert_eq!(syn_state.post_syn_nid, cmp_syn_state.post_syn_nid);
                assert_eq!(syn_state.conduction_delay, cmp_syn_state.conduction_delay);
                assert_approx_eq!(f32, syn_state.weight, cmp_syn_state.weight);
                assert_approx_eq!(
                    f32,
                    syn_state.short_term_stdp_offset,
                    cmp_syn_state.short_term_stdp_offset
                );
            }
        }
    }
}

#[test]
fn invariance_partitioning_buffering() {
    let thread_counts = vec![1, 6, 7];
    let mut instances = Vec::new();

    for thread_count in thread_counts {
        let mut params = get_scenario_template_params();

        params.technical_params.num_threads = Some(thread_count);
        instances.push(instance::create_instance(params).unwrap());
    }

    assert_equivalence(&mut instances, 102, true, true, true);
}

#[test]
fn invariance_zero_effect_projection() {
    let mut params = get_scenario_template_params();
    params.layers[0].num_neurons = 80;
    params.layers[1].num_neurons = 20;
    for conn in params.layer_connections.iter_mut() {
        conn.initial_syn_weight = InitialSynWeight::Constant(0.5);
        conn.connect_density = 1.0;
        conn.conduction_delay_max_random_part = 0;
    }

    let mut params_zero_effect_projections = params.clone();

    for (idx, conn_params) in params.layer_connections.iter().enumerate() {
        let mut zero_effect_conn = &mut params_zero_effect_projections.layer_connections[idx];
        zero_effect_conn.initial_syn_weight = InitialSynWeight::Constant(0.0);
        let mut projection_params = &mut zero_effect_conn.projection_params;
        projection_params.long_term_stdp_params = None;
        projection_params.short_term_stdp_params = None;
        params_zero_effect_projections
            .layer_connections
            .push(conn_params.clone());
    }

    let mut instances = vec![
        instance::create_instance(params).unwrap(),
        instance::create_instance(params_zero_effect_projections).unwrap(),
    ];

    assert_equivalence(&mut instances, 120, true, false, false);
}

#[test]
fn zero_vs_absent_long_term_stdp() {
    let mut params = get_scenario_template_params();

    params.layers[0].plasticity_modulation_params = None;

    params.layer_connections[0].projection_params.stp_params = StpParams::NoStp;

    params.layer_connections[0]
        .projection_params
        .short_term_stdp_params = None;

    params.layer_connections[0]
        .projection_params
        .long_term_stdp_params = Some(StdpParams {
        factor_pre_before_post: 0.0,
        tau_pre_before_post: 20.0,
        factor_pre_after_post: 0.0,
        tau_pre_after_post: 20.0,
    });

    let mut instances = Vec::new();
    instances.push(instance::create_instance(params.clone()).unwrap());
    params.layer_connections[0]
        .projection_params
        .long_term_stdp_params = None;
    instances.push(instance::create_instance(params).unwrap());
    assert_equivalence(&mut instances, 100, true, true, true);
}

#[test]
fn zero_vs_absent_short_term_stdp() {
    let mut params = get_scenario_template_params();

    params.layers[0].plasticity_modulation_params = None;

    params.layer_connections[0].projection_params.stp_params = StpParams::NoStp;

    params.layer_connections[0]
        .projection_params
        .long_term_stdp_params = None;

    params.layer_connections[0]
        .projection_params
        .short_term_stdp_params = Some(ShortTermStdpParams {
        stdp_params: StdpParams {
            factor_pre_before_post: 0.0,
            tau_pre_before_post: 20.0,
            factor_pre_after_post: 0.0,
            tau_pre_after_post: 20.0,
        },
        tau: 500.0,
    });

    let mut instances = Vec::new();
    instances.push(instance::create_instance(params.clone()).unwrap());
    params.layer_connections[0]
        .projection_params
        .short_term_stdp_params = None;
    instances.push(instance::create_instance(params).unwrap());
    assert_equivalence(&mut instances, 100, true, true, true);
}

#[test]
fn zero_vs_absent_plasticity_modulation() {
    let mut params = get_scenario_template_params();
    let mut instances = Vec::new();
    instances.push(instance::create_instance(params.clone()).unwrap());
    params.layers[0].plasticity_modulation_params = None;
    params.layer_connections[0]
        .projection_params
        .long_term_stdp_params = None;
    instances.push(instance::create_instance(params).unwrap());
    assert_equivalence(&mut instances, 110, false, true, true);
}

#[test]
fn state_snapshot() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 5;
    params.layers.push(layer.clone());
    layer.num_neurons = 2;
    params.layers.push(layer);

    let mut conn_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    conn_params.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    conn_params.projection_params.synapse_params.max_weight = 1.5;
    conn_params.initial_syn_weight = InitialSynWeight::Constant(0.5);
    params.layer_connections.push(conn_params.clone());
    conn_params.to_layer_id = 1;
    conn_params.initial_syn_weight = InitialSynWeight::Constant(1.0);
    params.layer_connections.push(conn_params);

    let mut instance = create_instance(params).unwrap();
    tick(&mut instance, &[0]);

    let tick_1_result = tick(&mut instance, &[]);

    assert_equal(tick_1_result.spiking_nids, [5, 6]);
    assert_equal(tick_1_result.spiking_out_channel_ids, [0, 1]);

    let state_snapshot = instance.extract_state_snapshot();
    for i in 1..5 {
        assert_eq!(state_snapshot.synapse_states[i].pre_syn_nid, 0);
        assert_eq!(state_snapshot.synapse_states[i].post_syn_nid, i);
        assert_approx_eq!(f32, state_snapshot.synapse_states[i].weight, 0.5);
    }

    for i in 25..27 {
        assert_eq!(state_snapshot.synapse_states[i].pre_syn_nid, 0);
        assert_eq!(state_snapshot.synapse_states[i].post_syn_nid, i - 20);
        assert_approx_eq!(f32, state_snapshot.synapse_states[i].weight, 1.1);
    }

    for pre_syn_nid in 1..5 {
        for post_syn_nid in 5..7 {
            let idx = 25 + 2 * pre_syn_nid + post_syn_nid - 5;
            assert_eq!(state_snapshot.synapse_states[idx].pre_syn_nid, pre_syn_nid);
            assert_eq!(
                state_snapshot.synapse_states[idx].post_syn_nid,
                post_syn_nid
            );
            assert_approx_eq!(f32, state_snapshot.synapse_states[idx].weight, 1.0);
        }
    }
}

#[test]
fn no_self_innervation() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 2;
    params.layers.push(layer.clone());
    layer.num_neurons = 5;
    params.layers.push(layer);

    let mut conn_params = LayerConnectionParams::defaults_for_layer_ids(1, 0);
    params.layer_connections.push(conn_params.clone());
    conn_params.to_layer_id = 1;
    conn_params.allow_self_innervation = false;
    params.layer_connections.push(conn_params);

    let mut instance = create_instance(params).unwrap();

    let state_snapshot = instance.extract_state_snapshot();

    for synapse_state in state_snapshot.synapse_states {
        assert_ne!(synapse_state.pre_syn_nid, synapse_state.post_syn_nid);
    }
}

#[test]
fn multiple_projections_same_layer_pair() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 3;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut conn = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    conn.initial_syn_weight = InitialSynWeight::Constant(0.5);

    conn.connect_width = 0.0;
    conn.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    params.layer_connections.push(conn.clone());

    conn.connect_width = 2.0;
    let mut stdp_params_modified = STDP_PARAMS.clone();
    stdp_params_modified.factor_pre_before_post = 0.2;
    conn.projection_params.long_term_stdp_params = Some(stdp_params_modified);
    params.layer_connections.push(conn.clone());

    conn.connect_width = 0.0;
    conn.projection_params.long_term_stdp_params = None;
    conn.projection_params.short_term_stdp_params = Some(ShortTermStdpParams {
        stdp_params: STDP_PARAMS,
        tau: 500.0,
    });

    params.layer_connections.push(conn);

    let mut instance = create_instance(params).unwrap();

    let mut tick_input = TickInput::default();
    tick_input.force_spiking_nids.push(1);

    instance.tick(&tick_input).unwrap();

    tick_input.reset();
    tick_input.force_spiking_nids.push(4);
    instance.tick(&tick_input).unwrap();

    instance.tick_no_input_until_inclusive(50);
    instance.tick_no_input();

    let syn_states_from_1 = instance
        .extract_state_snapshot()
        .synapse_states
        .into_iter()
        .filter(|syn_state| syn_state.pre_syn_nid == 1)
        .collect_vec();

    assert_eq!(syn_states_from_1.len(), 5);

    assert_eq!(syn_states_from_1[0].projection_id, 0);
    assert_approx_eq!(f32, syn_states_from_1[0].weight, 0.6);
    assert_approx_eq!(f32, syn_states_from_1[0].short_term_stdp_offset, 0.0);

    assert_eq!(syn_states_from_1[2].projection_id, 1);
    assert_eq!(syn_states_from_1[2].post_syn_nid, 4);
    assert_approx_eq!(f32, syn_states_from_1[2].weight, 0.7);
    assert_approx_eq!(f32, syn_states_from_1[2].short_term_stdp_offset, 0.0);

    assert_eq!(syn_states_from_1[4].projection_id, 2);
    assert_eq!(syn_states_from_1[4].post_syn_nid, 4);
    assert_approx_eq!(f32, syn_states_from_1[4].weight, 0.5);
    assert_approx_eq!(
        f32,
        syn_states_from_1[4].short_term_stdp_offset,
        0.1 * (-49f32 / 500.0).exp()
    );
}

fn compute_sc_hash_single(pre_syn_nid: usize, post_syn_nid: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_usize(pre_syn_nid);
    hasher.write_usize(post_syn_nid);
    hasher.finish()
}

fn compute_sc_hash_multi(pre_syn_nids: &[usize], post_syn_nid: usize) -> u64 {
    let mut hasher = DefaultHasher::new();

    for pre_syn_nid in pre_syn_nids {
        hasher.write_usize(*pre_syn_nid);
    }

    hasher.write_usize(post_syn_nid);
    hasher.finish()
}

#[test]
fn sc_hashes_single() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 4;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection.conduction_delay_position_distance_scale_factor = 1.0;

    params.layer_connections.push(connection);
    params.technical_params.num_threads = Some(2);

    let mut instance = create_instance(params).unwrap();
    instance
        .set_sc_mode(SCMode::Single { threshold: 0.0 })
        .unwrap();

    tick(&mut instance, &[1, 3]);

    let tick_2_result = instance.tick_no_input();
    assert_equal(tick_2_result.spiking_nids, [6]);

    let mut tick_input = TickInput::new();
    tick_input.force_spiking_nids.extend([4, 7]);

    instance.tick(&tick_input).unwrap();

    let sc_hashes = instance.flush_sc_hashes();

    let expected_sc_hashes = HashSet::from([
        compute_sc_hash_single(1, 4),
        compute_sc_hash_single(3, 4),
        compute_sc_hash_single(1, 6),
        compute_sc_hash_single(3, 6),
        compute_sc_hash_single(1, 7),
        compute_sc_hash_single(3, 7),
    ]);

    assert_eq!(sc_hashes, expected_sc_hashes);
    assert!(instance.flush_sc_hashes().is_empty());
}

#[test]
fn sc_hashes_multi() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 4;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection.conduction_delay_position_distance_scale_factor = 1.0;

    params.layer_connections.push(connection);
    params.technical_params.num_threads = Some(2);

    let mut instance = create_instance(params).unwrap();
    instance
        .set_sc_mode(SCMode::Multi { threshold: 0.0 })
        .unwrap();

    tick(&mut instance, &[1, 3]);

    let tick_2_result = instance.tick_no_input();
    assert_equal(tick_2_result.spiking_nids, [6]);

    let mut tick_input = TickInput::new();
    tick_input.force_spiking_nids.extend([4, 7]);

    instance.tick(&tick_input).unwrap();

    let sc_hashes = instance.flush_sc_hashes();

    let expected_sc_hashes = HashSet::from([
        compute_sc_hash_multi(&[], 1),
        compute_sc_hash_multi(&[], 3),
        compute_sc_hash_multi(&[1, 3], 4),
        compute_sc_hash_multi(&[1, 3], 6),
        compute_sc_hash_multi(&[3, 1], 7),
    ]);

    assert_eq!(sc_hashes, expected_sc_hashes);
    assert!(instance.flush_sc_hashes().is_empty());
}

#[test]
fn no_sc_hashes_inhibitory() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection
        .projection_params
        .synapse_params
        .weight_scale_factor = -1.0;

    params.layer_connections.push(connection);

    let mut instance = create_instance(params).unwrap();
    instance
        .set_sc_mode(SCMode::Multi { threshold: 0.0 })
        .unwrap();

    tick(&mut instance, &[0]);
    let mut tick_input = TickInput::default();
    tick_input.force_spiking_nids.push(1);
    instance.tick(&tick_input).unwrap();

    let sc_hashes = instance.flush_sc_hashes();
    let expected_sc_hashes =
        HashSet::from([compute_sc_hash_multi(&[], 0), compute_sc_hash_multi(&[], 1)]);

    assert_eq!(sc_hashes, expected_sc_hashes);

    instance
        .set_sc_mode(SCMode::Single { threshold: 0.0 })
        .unwrap();

    tick(&mut instance, &[0]);
    let mut tick_input = TickInput::default();
    tick_input.force_spiking_nids.push(1);
    instance.tick(&tick_input).unwrap();

    let sc_hashes = instance.flush_sc_hashes();

    assert!(sc_hashes.is_empty());
}

#[test]
fn sc_hashes_threshold_single() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection.initial_syn_weight = InitialSynWeight::Constant(1.0);
    connection.projection_params.synapse_params.max_weight = 1.1;
    connection.projection_params.long_term_stdp_params = Some(STDP_PARAMS);

    params.layer_connections.push(connection);

    let mut instance = create_instance(params).unwrap();
    instance
        .set_sc_mode(SCMode::Single { threshold: 1.0 })
        .unwrap();

    tick(&mut instance, &[0]);
    instance.tick_no_input();

    let sc_hashes = instance.flush_sc_hashes();

    assert!(sc_hashes.is_empty());

    instance.reset_ephemeral_state();

    tick(&mut instance, &[0]);
    instance.tick_no_input();

    let sc_hashes = instance.flush_sc_hashes();
    let expected_sc_hashes = HashSet::from([compute_sc_hash_single(0, 1)]);

    assert_eq!(sc_hashes, expected_sc_hashes);
}

#[test]
fn sc_hashes_threshold_multi() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 1;
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection.initial_syn_weight = InitialSynWeight::Constant(1.0);
    connection.projection_params.synapse_params.max_weight = 1.1;
    connection.projection_params.long_term_stdp_params = Some(STDP_PARAMS);

    params.layer_connections.push(connection);

    let mut instance = create_instance(params).unwrap();
    instance
        .set_sc_mode(SCMode::Multi { threshold: 1.0 })
        .unwrap();

    tick(&mut instance, &[0]);
    instance.tick_no_input();

    let sc_hashes = instance.flush_sc_hashes();
    let expected_sc_hashes =
        HashSet::from([compute_sc_hash_multi(&[], 0), compute_sc_hash_multi(&[], 1)]);

    assert_eq!(sc_hashes, expected_sc_hashes);

    instance.reset_ephemeral_state();

    tick(&mut instance, &[0]);
    instance.tick_no_input();

    let sc_hashes = instance.flush_sc_hashes();
    let expected_sc_hashes = HashSet::from([
        compute_sc_hash_multi(&[], 0),
        compute_sc_hash_multi(&[0], 1),
    ]);

    assert_eq!(sc_hashes, expected_sc_hashes);
}

#[test]
fn sc_hashes_ephemeral_state_reset() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();

    layer.num_neurons = 4;
    params.layers.push(layer);

    let mut instance = create_instance(params).unwrap();
    instance
        .set_sc_mode(SCMode::Multi { threshold: 0.0 })
        .unwrap();

    tick(&mut instance, &[1, 3]);

    instance.reset_ephemeral_state();

    assert!(instance.flush_sc_hashes().is_empty());
}

#[test]
fn para_spikes_potentiation_no_depression() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_01.projection_params.synapse_params.max_weight = 0.5;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut connection_12 = LayerConnectionParams::defaults_for_layer_ids(1, 2);
    connection_12.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_12.projection_params.synapse_params.max_weight = 1.0;
    connection_12.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_12.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_12);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 2]);

    instance.tick_no_input_until_exclusive(4);

    let mut tick_input = TickInput::new();
    tick_input.force_spiking_nids.push(7);
    instance.tick(&tick_input).unwrap();

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    assert_approx_eq!(f32, weights[0], 0.1); // 0 -> 3
    assert_approx_eq!(f32, weights[1], 0.2); // 0 -> 4
    assert_approx_eq!(f32, weights[2], 0.1); // 0 -> 5

    assert_approx_eq!(f32, weights[3], 0.1); // 1 -> 3
    assert_approx_eq!(f32, weights[4], 0.1); // 1 -> 4
    assert_approx_eq!(f32, weights[5], 0.1); // 1 -> 5

    assert_approx_eq!(f32, weights[6], 0.1); // 2 -> 3
    assert_approx_eq!(f32, weights[7], 0.2); // 2 -> 4
    assert_approx_eq!(f32, weights[8], 0.1); // 2 -> 5

    assert_approx_eq!(f32, weights[9], 0.1); // 3 -> 6
    assert_approx_eq!(f32, weights[10], 0.1); // 3 -> 7
    assert_approx_eq!(f32, weights[11], 0.1); // 3 -> 8

    assert_approx_eq!(f32, weights[12], 0.1); // 4 -> 6
    assert_approx_eq!(f32, weights[13], 0.1); // 4 -> 7
    assert_approx_eq!(f32, weights[14], 0.1); // 4 -> 8

    assert_approx_eq!(f32, weights[15], 0.1); // 5 -> 6
    assert_approx_eq!(f32, weights[16], 0.1); // 5 -> 7
    assert_approx_eq!(f32, weights[17], 0.1); // 5 -> 8
}

#[test]
fn para_spikes_potentiation_and_depression() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_01.projection_params.synapse_params.max_weight = 0.4;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut connection_12 = LayerConnectionParams::defaults_for_layer_ids(1, 2);
    connection_12.initial_syn_weight = InitialSynWeight::Constant(0.2);
    connection_12.projection_params.synapse_params.max_weight = 0.4;
    connection_12.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_12.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_12);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 2]);

    instance.tick_no_input_until_exclusive(4);

    let mut tick_input = TickInput::new();
    tick_input.force_spiking_nids.push(4);
    tick_input.force_spiking_nids.push(7);
    instance.tick(&tick_input).unwrap();

    instance.tick_no_input_for(3);

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    let expected_potentiated_weight = 0.1 + 0.1 * (-1.0 / 20.0f32).exp();
    let expected_depressed_weight = 0.2 - 0.11 * (-1.0 / 25.0f32).exp();

    assert_approx_eq!(f32, weights[0], 0.1); // 0 -> 3
    assert_approx_eq!(f32, weights[1], expected_potentiated_weight); // 0 -> 4
    assert_approx_eq!(f32, weights[2], 0.1); // 0 -> 5

    assert_approx_eq!(f32, weights[3], 0.1); // 1 -> 3
    assert_approx_eq!(f32, weights[4], 0.1); // 1 -> 4
    assert_approx_eq!(f32, weights[5], 0.1); // 1 -> 5

    assert_approx_eq!(f32, weights[6], 0.1); // 2 -> 3
    assert_approx_eq!(f32, weights[7], expected_potentiated_weight); // 2 -> 4
    assert_approx_eq!(f32, weights[8], 0.1); // 2 -> 5

    assert_approx_eq!(f32, weights[9], 0.2); // 3 -> 6
    assert_approx_eq!(f32, weights[10], 0.2); // 3 -> 7
    assert_approx_eq!(f32, weights[11], 0.2); // 3 -> 8

    assert_approx_eq!(f32, weights[12], 0.2); // 4 -> 6
    assert_approx_eq!(f32, weights[13], expected_depressed_weight); // 4 -> 7
    assert_approx_eq!(f32, weights[14], 0.2); // 4 -> 8

    assert_approx_eq!(f32, weights[15], 0.2); // 5 -> 6
    assert_approx_eq!(f32, weights[16], 0.2); // 5 -> 7
    assert_approx_eq!(f32, weights[17], 0.2); // 5 -> 8
}

#[test]
fn para_spike_and_normal_spike_simultaneous() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.5);
    connection_01.projection_params.synapse_params.max_weight = 0.8;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 2]);

    instance.tick_no_input();
    let tick_result = instance.tick_no_input();
    assert_equal(tick_result.spiking_nids, [4]);

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    assert_approx_eq!(f32, weights[0], 0.5); // 0 -> 3
    assert_approx_eq!(f32, weights[1], 0.6); // 0 -> 4
    assert_approx_eq!(f32, weights[2], 0.5); // 0 -> 5

    assert_approx_eq!(f32, weights[3], 0.5); // 1 -> 3
    assert_approx_eq!(f32, weights[4], 0.5); // 1 -> 4
    assert_approx_eq!(f32, weights[5], 0.5); // 1 -> 5

    assert_approx_eq!(f32, weights[6], 0.5); // 2 -> 3
    assert_approx_eq!(f32, weights[7], 0.6); // 2 -> 4
    assert_approx_eq!(f32, weights[8], 0.5); // 2 -> 5
}

#[test]
fn para_spike_then_normal_spike() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.4);
    connection_01.projection_params.synapse_params.max_weight = 0.8;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 2]);
    instance.tick_no_input();
    tick(&mut instance, &[1]);

    let tick_result = instance.tick_no_input();
    assert_equal(tick_result.spiking_nids, [4]);

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    assert_approx_eq!(f32, weights[1], 0.5); // 0 -> 4
    assert_approx_eq!(f32, weights[4], 0.4); // 1 -> 4
    assert_approx_eq!(f32, weights[7], 0.5); // 2 -> 4
}

#[test]
fn para_spike_then_force_spike() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_01.projection_params.synapse_params.max_weight = 0.5;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 2]);

    instance.tick_no_input_for(2);

    let mut tick_input = TickInput::new();
    tick_input.force_spiking_nids.push(4);
    instance.tick(&tick_input).unwrap();

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    assert_approx_eq!(f32, weights[0], 0.1); // 0 -> 3
    assert_approx_eq!(f32, weights[1], 0.2); // 0 -> 4
    assert_approx_eq!(f32, weights[2], 0.1); // 0 -> 5

    assert_approx_eq!(f32, weights[3], 0.1); // 1 -> 3
    assert_approx_eq!(f32, weights[4], 0.1); // 1 -> 4
    assert_approx_eq!(f32, weights[5], 0.1); // 1 -> 5

    assert_approx_eq!(f32, weights[6], 0.1); // 2 -> 3
    assert_approx_eq!(f32, weights[7], 0.2); // 2 -> 4
    assert_approx_eq!(f32, weights[8], 0.1); // 2 -> 5
}

#[test]
fn force_spike_then_para_spike() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_01.projection_params.synapse_params.max_weight = 0.5;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut instance = create_instance(params).unwrap();

    let mut tick_input = TickInput::new();
    tick_input.force_spiking_nids.push(4);
    instance.tick(&tick_input).unwrap();

    instance.tick_no_input_for(7);

    tick(&mut instance, &[0, 2]);

    instance.tick_no_input_for(2);

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    let expected_depression = 0.11 * (-10.0 / 25.0f32).exp();
    let expected_potentiation = 0.1;
    let expected_changed_weight = 0.1 - expected_depression + expected_potentiation;

    assert_approx_eq!(f32, weights[0], 0.1); // 0 -> 3
    assert_approx_eq!(f32, weights[1], expected_changed_weight); // 0 -> 4
    assert_approx_eq!(f32, weights[2], 0.1); // 0 -> 5

    assert_approx_eq!(f32, weights[3], 0.1); // 1 -> 3
    assert_approx_eq!(f32, weights[4], 0.1); // 1 -> 4
    assert_approx_eq!(f32, weights[5], 0.1); // 1 -> 5

    assert_approx_eq!(f32, weights[6], 0.1); // 2 -> 3
    assert_approx_eq!(f32, weights[7], expected_changed_weight); // 2 -> 4
    assert_approx_eq!(f32, weights[8], 0.1); // 2 -> 5
}

#[test]
fn repeated_para_spikes_lead_to_normal_spikes() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.0);
    connection_01.projection_params.synapse_params.max_weight = 0.501;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut instance = create_instance(params).unwrap();

    for _ in 0..5 {
        tick(&mut instance, &[0, 2]);
        instance.tick_no_input();
        let tick_result = instance.tick_no_input();
        assert!(tick_result.spiking_nids.is_empty());
        instance.tick_no_input_for(10);
    }

    tick(&mut instance, &[0, 2]);
    instance.tick_no_input();
    let tick_result = instance.tick_no_input();
    assert_equal(&tick_result.spiking_nids, &[4]);

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    assert_approx_eq!(f32, weights[1], 0.501); // 0 -> 4
    assert_approx_eq!(f32, weights[4], 0.0); // 1 -> 4
    assert_approx_eq!(f32, weights[7], 0.501); // 2 -> 4
}

#[test]
fn para_spikes_ephemeral_state_reset() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 3;
    layer.use_para_spikes = true;

    params.layers.push(layer.clone());
    params.layers.push(layer);

    let mut connection_01 = LayerConnectionParams::defaults_for_layer_ids(0, 1);
    connection_01.initial_syn_weight = InitialSynWeight::Constant(0.1);
    connection_01.projection_params.synapse_params.max_weight = 0.5;
    connection_01.projection_params.long_term_stdp_params = Some(STDP_PARAMS);
    connection_01.conduction_delay_position_distance_scale_factor = 2.0;
    params.layer_connections.push(connection_01);

    let mut instance = create_instance(params).unwrap();

    tick(&mut instance, &[0, 2]);

    instance.reset_ephemeral_state();

    instance.tick_no_input_until_exclusive(4);

    let weights = instance
        .extract_state_snapshot()
        .synapse_states
        .iter()
        .map(|syn_state| syn_state.weight)
        .collect_vec();

    for weight in weights {
        assert_approx_eq!(f32, weight, 0.1);
    }
}
