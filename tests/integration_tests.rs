use float_cmp::assert_approx_eq;
use itertools::{assert_equal, Itertools};
use morphine::{
    instance::{self, create_instance, Instance},
    params::{
        InitialSynWeight, InstanceParams, LayerConnectionParams, LayerParams,
        PlasticityModulationParams, ShortTermStdpParams, StdpParams, StpParams,
    },
};
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng,
};

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
fn single_neuron() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.num_neurons = 1;

    params.layers.push(layer);

    let mut instance = create_instance(params).unwrap();

    // t = 0: empty context, expect empty result
    let tick_0_result = instance.tick(&[], 0.0, false);
    assert!(tick_0_result.out_spiking_channel_ids.is_empty());

    // t = 1: input channel 0 ticks, expect output channel 0 tick
    let tick_1_result = instance.tick(&[0], 0.0, false);
    assert_equal(tick_1_result.out_spiking_channel_ids, [0]);
}

#[test]
fn single_direct_mapped_output() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    // double spike on nid 0 ...
    let tick_0_result = instance.tick(&[0, 0], 0.0, false);
    assert!(tick_0_result.out_spiking_channel_ids.is_empty());

    // ... leads to spike at nid 1 at next tick, mapped to output channel 0
    let tick_1_result = instance.tick_no_input();
    assert_equal(tick_1_result.out_spiking_channel_ids, [0]);
}

#[test]
fn missed_spike_after_leakage() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    // first spike on nid 0
    let tick_0_result = instance.tick(&[0], 0.0, false);
    assert!(tick_0_result.out_spiking_channel_ids.is_empty());

    // seconds spike on nid 0
    let tick_1_result = instance.tick(&[0], 0.0, false);
    assert!(tick_1_result.out_spiking_channel_ids.is_empty());

    // spike is missed, but voltaged is close to threshold
    let tick_2_result = instance.tick(&[], 0.0, true);
    assert!(tick_2_result.out_spiking_channel_ids.is_empty());

    assert_approx_eq!(
        f32,
        tick_2_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5 * (-1.0 / 10.0f32).exp() + 0.5
    );
}

#[test]
fn voltage_trajectory() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    instance.tick(&[0], 0.0, false);

    let tick_1_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5
    );

    for _ in 0..4 {
        instance.tick_no_input();
    }

    let tick_6_result = instance.tick(&[], 0.0, true);

    assert_approx_eq!(
        f32,
        tick_6_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5 * (-5.0 / 10.0f32).exp()
    );
}

#[test]
fn no_psp_during_refractory_period() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    // make neuron 1 fire one tick later
    instance.tick(&[0, 0], 0.0, false);

    let tick_1_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.0
    );

    instance.tick(&[0], 0.0, false);

    let tick_3_result = instance.tick(&[], 0.0, true);

    // neuron 1 is in refractory period
    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.0
    );

    while instance.get_tick_period() < 9 {
        instance.tick_no_input();
    }

    // too early to cause a psp
    instance.tick(&[0], 0.0, false);

    let tick_10_result = instance.tick(&[0], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_10_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.0
    );

    let tick_11_result = instance.tick(&[], 0.0, true);

    // neuron 1 is not in refractory period anymore
    assert_approx_eq!(
        f32,
        tick_11_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5
    );
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
    instance.tick(&[0, 0], 0.0, false);

    let tick_2_result = instance.tick(&[], 0.0, false);

    assert_equal(tick_2_result.spiking_nids, [1]);
    assert_equal(tick_2_result.out_spiking_channel_ids, []);

    let tick_3_result = instance.tick(&[], 0.0, true);

    assert!(tick_3_result.spiking_nids.is_empty());
    assert!(tick_3_result.out_spiking_channel_ids.is_empty());
    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.unwrap().neuron_states[2].voltage,
        0.9
    );
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
    instance.tick(&[0, 0], 0.0, false);

    let tick_1_result = instance.tick(&[], 0.0, true);

    // voltage floored at -0.6
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[1].voltage,
        -0.6
    );
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
    instance.tick(&[0, 0], 0.0, false);

    // nid 1 spikes, threshold drops to 0.4
    instance.tick_no_input();

    // now a psp of 0.5 is sufficient to spike
    instance.tick(&[0], 0.0, false);
    let tick_3_result = instance.tick_no_input();
    assert_equal(tick_3_result.out_spiking_channel_ids, [0]);

    instance.tick_no_input();
    instance.tick_no_input();

    // threshold offset decayed and psp of 0.5 is not sufficient anymore
    instance.tick(&[0], 0.0, false);
    let tick_7_result = instance.tick_no_input();
    assert!(tick_7_result.out_spiking_channel_ids.is_empty());
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

    instance.tick(&[0], 0.0, false);
    instance.tick_no_input(); // nid 1 spikes
    instance.tick(&[0], 0.0, false);
    let tick_3_result = instance.tick(&[], 0.0, true);

    assert!(tick_3_result.out_spiking_channel_ids.is_empty());

    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.unwrap().neuron_states[1].voltage,
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

    instance.tick(&[0], 0.0, false);
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

    instance.tick(&[0], 0.0, false);
    instance.tick_no_input(); // nid 1 spikes
    instance.tick_no_input();

    instance.tick(&[0], 0.0, false);

    let tick_3_result = instance.tick(&[], 0.0, true);

    assert!(tick_3_result.out_spiking_channel_ids.is_empty());

    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.unwrap().neuron_states[1].voltage,
        1.0 + 0.1 * (-3.0f32).exp()
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

    instance.tick(&[0], 0.0, false);

    instance.tick(&[1], 0.0, false);
    instance.tick_no_input_until(3);
    instance.tick(&[1], 0.0, false);

    instance.tick_no_input_until(20);
    instance.tick(&[0], 0.0, false);

    let tick_21_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_21_result.state_snapshot.unwrap().neuron_states[1].voltage,
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

    instance.tick(&[0], 0.0, false);

    for _tick_period in 1..5 {
        instance.tick_no_input();
    }

    instance.tick(&[1], 0.0, false);
    instance.tick_no_input(); // psp arrives at tick 6, contributes to stdp
    instance.tick(&[1], 0.0, false);
    instance.tick_no_input(); // psp arrives at tick 8, does not contribute to stdp
    instance.tick(&[1], 0.0, false);

    let tick_10_result = instance.tick(&[], 0.0, true);

    let expected_lt_stdp_value_tick_6 = -0.11 * (-6.0 / 25.0f32).exp();
    let expected_st_stdp_value_tick_6 = -0.06 * (-6.0 / 20.0f32).exp();

    let expected_weight = 0.4 + expected_lt_stdp_value_tick_6;
    let expected_st_stdp_offset_tick_10 = expected_st_stdp_value_tick_6 * (-4.0 / 10.0f32).exp();

    let expected_psp = expected_weight + expected_st_stdp_offset_tick_10;

    assert_approx_eq!(
        f32,
        tick_10_result.state_snapshot.unwrap().neuron_states[0].voltage,
        expected_psp
    );
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

    instance.tick(&[1], 0.0, false);
    instance.tick_no_input_until(8);
    instance.tick(&[0], 0.0, false); // psp at t = 11
    instance.tick_no_input_until(12);
    instance.tick(&[1], 0.0, false);
    instance.tick_no_input_until(21);
    instance.tick(&[0], 0.0, false); // psp at t = 24
    instance.tick_no_input_until(30);
    instance.tick(&[0], 0.0, false); // psp at t = 33
    instance.tick_no_input_until(33);

    let tick_33_result = instance.tick(&[], 0.0, true);

    let tick_11_stdp = -0.11 * (-11.0 / 25.0f32).exp();
    let tick_12_stdp = 0.1 * (-1.0 / 20.0f32).exp();
    let tick_24_stdp = -0.11 * (-12.0 / 25.0f32).exp();
    let expected_weight = 0.4 + tick_11_stdp + tick_12_stdp + tick_24_stdp;

    assert_approx_eq!(
        f32,
        tick_33_result.state_snapshot.unwrap().neuron_states[1].voltage,
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

    instance.tick(&[0], 0.0, false);
    instance.tick_no_input_until(8);

    //this spike will arrive at t = 13, after the post-synaptic spike -> depression
    instance.tick(&[5], 0.0, false);

    instance.tick_no_input_until(10);
    instance.tick(&[9], 0.0, false);

    // spikes from neuron 0 and 9 should arrive at neuron 19 (output channel 9) simultaneously
    let tick_11_result = instance.tick_no_input();
    assert_equal(tick_11_result.out_spiking_channel_ids, [9]);

    instance.tick_no_input_until(25);

    instance.tick(&[9], 0.0, false);

    let tick_26_result = instance.tick(&[9], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_26_result.state_snapshot.unwrap().neuron_states[19].voltage,
        0.6 // synapse potentiated
    );

    // make nid 19 spike to bring it back to reset voltage
    let tick_27_result = instance.tick(&[9], 0.0, false);
    assert_equal(tick_27_result.out_spiking_channel_ids, [9]);

    // this will cause a psp of psp due to potentiation
    instance.tick(&[0], 0.0, false);

    // cause as spike at nid 19 later on to bring it back to reset voltage
    instance.tick(&[0], 0.0, false);

    instance.tick_no_input_until(39);

    let tick_39_result = instance.tick(&[], 0.0, true);

    assert_approx_eq!(
        f32,
        tick_39_result.state_snapshot.unwrap().neuron_states[19].voltage,
        0.6
    );

    let tick_40_result = instance.tick_no_input();
    assert_equal(tick_40_result.out_spiking_channel_ids, [9]);

    instance.tick_no_input_until(50);
    instance.tick(&[5], 0.0, false);
    instance.tick_no_input_until(55);

    let tick_55_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_55_result.state_snapshot.unwrap().neuron_states[19].voltage,
        0.5 - 0.11 * (-2.0 / 25.0f32).exp() // synapse depressed
    );

    // again: induce spike to get back to reset voltage
    instance.tick(&[9, 9], 0.0, false);
    let tick_57_result = instance.tick_no_input();
    assert_equal(tick_57_result.out_spiking_channel_ids, [9]);

    instance.tick_no_input_until(70);

    // psp will arrive at t = 74
    instance.tick(&[6], 0.0, false);
    instance.tick_no_input_until(74);

    let tick_74_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_74_result.state_snapshot.unwrap().neuron_states[19].voltage,
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

    instance.tick(&[0, 0], 0.0, false);
    instance.tick_no_input(); // nid 1 spikes
    instance.tick_no_input_until(1500);
    instance.tick(&[0], 0.0, false);

    let tick_1501_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_1501_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5 // unchanged synapse
    );
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
    instance.tick(&[0, 0], 0.0, false);

    instance.tick_no_input_until(53);

    // dopamine released at t = 60
    instance.tick(&[], 1.5, false);

    instance.tick_no_input_until(1500);
    instance.tick(&[0], 0.0, false);

    let tick_1501_result = instance.tick(&[], 0.0, true);

    let stdp_value = 0.2; // 2 * 0.1 (two transmissions at same synapse)
    let elig_trace_value = 1.5 * (-59.0 / 1000f32).exp();

    assert_approx_eq!(
        f32,
        tick_1501_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5 + 0.3 * stdp_value * elig_trace_value
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

    instance.tick(&[0], 0.0, false);

    let tick_1_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[2].voltage,
        0.5 * 0.8
    );

    instance.tick_no_input_until(6);

    let tick_6_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_6_result.state_snapshot.unwrap().neuron_states[3].voltage,
        0.5 * 0.8
    );

    instance.tick_no_input_until(500); // let voltages decay to near zero

    instance.tick(&[1], 0.0, false);
    let tick_8_result = instance.tick(&[], 0.0, true);
    assert_approx_eq!(
        f32,
        tick_8_result.state_snapshot.unwrap().neuron_states[3].voltage,
        0.5 * 0.8
    );

    instance.tick_no_input_until(1000);

    // both outgoing synapses of nid 0 are still depressed
    instance.tick(&[0], 0.0, false);
    let tick_1001_result = instance.tick(&[], 0.0, true);

    let expected_stp_factor = 0.8 * (1.0 - 0.6 * (-1000.0 / 800.0f32).exp());

    assert_approx_eq!(
        f32,
        tick_1001_result.state_snapshot.unwrap().neuron_states[2].voltage,
        0.5 * expected_stp_factor
    );

    instance.tick_no_input_until(1006);

    let tick_1006_result = instance.tick(&[], 0.0, true);

    assert_approx_eq!(
        f32,
        tick_1006_result.state_snapshot.unwrap().neuron_states[3].voltage,
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

fn assert_equivalence(instances: &mut [Instance], t_stop: usize) {
    let all_in_channels: Vec<usize> = (0..800).collect();
    let mut rng = StdRng::seed_from_u64(0);
    let reward_dist = Uniform::new(0.0, 0.005);

    for _ in 0..t_stop {
        let spiking_channels = all_in_channels
            .choose_multiple(&mut rng, 5)
            .copied()
            .collect::<Vec<_>>();

        let reward = reward_dist.sample(&mut rng);

        let mut tick_results = instances
            .iter_mut()
            .map(|instance| instance.tick(&spiking_channels, reward, false))
            .collect_vec();

        let cmp_result = tick_results.pop().unwrap();

        for tick_result in tick_results {
            assert_eq!(
                tick_result.out_spiking_channel_ids,
                cmp_result.out_spiking_channel_ids
            );
            assert_eq!(tick_result.spiking_nids, cmp_result.spiking_nids);
            assert_eq!(
                tick_result.synaptic_transmission_count,
                cmp_result.synaptic_transmission_count
            );
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

    assert_equivalence(&mut instances, 102);
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
    assert_equivalence(&mut instances, 100);
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
    assert_equivalence(&mut instances, 100);
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
    assert_equivalence(&mut instances, 110);
}
