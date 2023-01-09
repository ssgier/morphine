use float_cmp::assert_approx_eq;
use itertools::{assert_equal, Itertools};
use morphine::{
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

fn tick_extract_snapshot(instance: &mut Instance, in_channel_ids: &[usize]) -> TickResult {
    let mut tick_input = TickInput::from_spiking_in_channel_ids(in_channel_ids);
    tick_input.extract_state_snapshot = true;
    instance.tick(&tick_input).unwrap()
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
    let tick_2_result = tick_extract_snapshot(&mut instance, &[]);
    assert!(tick_2_result.spiking_out_channel_ids.is_empty());

    assert_approx_eq!(
        f32,
        tick_2_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5 * (-1.0 / 10.0f32).exp() + 0.5
    );
}

#[test]
fn voltage_trajectory() {
    let mut instance = make_simple_1_in_1_out_instance(0.5);

    tick(&mut instance, &[0]);

    let tick_1_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.5
    );

    for _ in 0..4 {
        instance.tick_no_input();
    }

    let tick_6_result = tick_extract_snapshot(&mut instance, &[]);

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
    tick(&mut instance, &[0, 0]);

    let tick_1_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.0
    );

    tick(&mut instance, &[0]);

    let tick_3_result = tick_extract_snapshot(&mut instance, &[]);

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
    tick(&mut instance, &[0]);

    let tick_10_result = tick_extract_snapshot(&mut instance, &[0]);
    assert_approx_eq!(
        f32,
        tick_10_result.state_snapshot.unwrap().neuron_states[1].voltage,
        0.0
    );

    let tick_11_result = tick_extract_snapshot(&mut instance, &[]);

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
    tick(&mut instance, &[0, 0]);

    let tick_2_result = tick(&mut instance, &[]);

    assert_equal(tick_2_result.spiking_nids, [1]);
    assert_equal(tick_2_result.spiking_out_channel_ids, []);

    let tick_3_result = tick_extract_snapshot(&mut instance, &[]);

    assert!(tick_3_result.spiking_nids.is_empty());
    assert!(tick_3_result.spiking_out_channel_ids.is_empty());
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
    tick(&mut instance, &[0, 0]);

    let tick_1_result = tick_extract_snapshot(&mut instance, &[]);

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
    let tick_2_result = tick_extract_snapshot(&mut instance, &[0]);
    assert_approx_eq!(
        f32,
        tick_2_result.state_snapshot.unwrap().synapse_states[0].weight,
        1.1
    );

    let tick_3_result = tick_extract_snapshot(&mut instance, &[]);

    assert!(tick_3_result.spiking_out_channel_ids.is_empty());

    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.as_ref().unwrap().neuron_states[1].voltage,
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

    let tick_3_result = tick_extract_snapshot(&mut instance, &[]);

    assert!(tick_3_result.spiking_out_channel_ids.is_empty());

    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.as_ref().unwrap().neuron_states[1].voltage,
        1.0 + 0.1 * (-3.0f32).exp()
    );

    assert_approx_eq!(
        f32,
        tick_3_result.state_snapshot.unwrap().synapse_states[0].weight,
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
    instance.tick_no_input_until(3);
    tick(&mut instance, &[1]);

    instance.tick_no_input_until(20);
    let tick_20_result = tick_extract_snapshot(&mut instance, &[0]);

    assert_eq!(
        tick_20_result
            .state_snapshot
            .as_ref()
            .unwrap()
            .synapse_states[1]
            .pre_syn_nid,
        0
    );
    assert_eq!(
        tick_20_result
            .state_snapshot
            .as_ref()
            .unwrap()
            .synapse_states[1]
            .post_syn_nid,
        1
    );

    assert_approx_eq!(
        f32,
        tick_20_result.state_snapshot.unwrap().synapse_states[1].weight,
        0.5 // synapse potentiated, but only once
    );

    let tick_21_result = tick_extract_snapshot(&mut instance, &[]);
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

    tick(&mut instance, &[0]);

    for _tick_period in 1..5 {
        instance.tick_no_input();
    }

    tick(&mut instance, &[1]);
    instance.tick_no_input(); // psp arrives at tick 6, contributes to stdp
    tick(&mut instance, &[1]);
    instance.tick_no_input(); // psp arrives at tick 8, does not contribute to stdp
    tick(&mut instance, &[1]);

    let tick_10_result = tick_extract_snapshot(&mut instance, &[]);

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

    tick(&mut instance, &[1]);
    instance.tick_no_input_until(8);
    tick(&mut instance, &[0]); // psp at t = 11
    instance.tick_no_input_until(12);
    tick(&mut instance, &[1]);
    instance.tick_no_input_until(21);
    tick(&mut instance, &[0]); // psp at t = 24
    instance.tick_no_input_until(30);
    tick(&mut instance, &[0]); // psp at t = 33
    instance.tick_no_input_until(33);

    let tick_33_result = tick_extract_snapshot(&mut instance, &[]);

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

    tick(&mut instance, &[0]);
    instance.tick_no_input_until(8);

    //this spike will arrive at t = 13, after the post-synaptic spike -> depression
    tick(&mut instance, &[5]);

    instance.tick_no_input_until(10);
    tick(&mut instance, &[9]);

    // spikes from neuron 0 and 9 should arrive at neuron 19 (output channel 9) simultaneously
    let tick_11_result = instance.tick_no_input();
    assert_equal(tick_11_result.spiking_out_channel_ids, [9]);

    instance.tick_no_input_until(25);

    tick(&mut instance, &[9]);

    let tick_26_result = tick_extract_snapshot(&mut instance, &[9]);
    assert_approx_eq!(
        f32,
        tick_26_result.state_snapshot.unwrap().neuron_states[19].voltage,
        0.6 // synapse potentiated
    );

    // make nid 19 spike to bring it back to reset voltage
    let tick_27_result = tick(&mut instance, &[9]);
    assert_equal(tick_27_result.spiking_out_channel_ids, [9]);

    // this will cause a psp of psp due to potentiation
    tick(&mut instance, &[0]);

    // cause as spike at nid 19 later on to bring it back to reset voltage
    tick(&mut instance, &[0]);

    instance.tick_no_input_until(39);

    let tick_39_result = tick_extract_snapshot(&mut instance, &[]);

    assert_approx_eq!(
        f32,
        tick_39_result.state_snapshot.unwrap().neuron_states[19].voltage,
        0.6
    );

    let tick_40_result = instance.tick_no_input();
    assert_equal(tick_40_result.spiking_out_channel_ids, [9]);

    instance.tick_no_input_until(50);
    tick(&mut instance, &[5]);
    instance.tick_no_input_until(55);

    let tick_55_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_55_result.state_snapshot.unwrap().neuron_states[19].voltage,
        0.5 - 0.11 * (-2.0 / 25.0f32).exp() // synapse depressed
    );

    // again: induce spike to get back to reset voltage
    tick(&mut instance, &[9, 9]);
    let tick_57_result = instance.tick_no_input();
    assert_equal(tick_57_result.spiking_out_channel_ids, [9]);

    instance.tick_no_input_until(70);

    // psp will arrive at t = 74
    tick(&mut instance, &[6]);
    instance.tick_no_input_until(74);

    let tick_74_result = tick_extract_snapshot(&mut instance, &[]);
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

    tick(&mut instance, &[0, 0]);
    instance.tick_no_input(); // nid 1 spikes
    instance.tick_no_input_until(1500);
    tick(&mut instance, &[0]);

    let tick_1501_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_1501_result.state_snapshot.unwrap().neuron_states[1].voltage,
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

    let mut tick_input = TickInput::from_reward(-1.0);
    tick_input.extract_state_snapshot = true;
    let tick_result = instance.tick(&tick_input).unwrap();

    let expected_depression = 0.1 * (-1.0 / 15.0f32).exp();

    let synapse_states = tick_result.state_snapshot.unwrap().synapse_states;
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

    instance.tick_no_input_until(53);

    // dopamine released at t = 60
    instance.tick(&TickInput::from_reward(1.5)).unwrap();

    instance.tick_no_input_until(1500);

    let tick_1500_result = tick_extract_snapshot(&mut instance, &[0]);

    let stdp_value = 0.2; // 2 * 0.1 (two transmissions at same synapse)
    let elig_trace_value = 1.5 * (-59.0 / 1000f32).exp();
    let expected_weight = 0.5 + 0.3 * stdp_value * elig_trace_value;

    assert_approx_eq!(
        f32,
        tick_1500_result.state_snapshot.unwrap().synapse_states[0].weight,
        expected_weight
    );

    let tick_1501_result = tick_extract_snapshot(&mut instance, &[]);

    assert_approx_eq!(
        f32,
        tick_1501_result.state_snapshot.unwrap().neuron_states[1].voltage,
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

    let tick_1_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_1_result.state_snapshot.unwrap().neuron_states[2].voltage,
        0.5 * 0.8
    );

    instance.tick_no_input_until(6);

    let tick_6_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_6_result.state_snapshot.unwrap().neuron_states[3].voltage,
        0.5 * 0.8
    );

    instance.tick_no_input_until(500); // let voltages decay to near zero

    tick(&mut instance, &[1]);
    let tick_8_result = tick_extract_snapshot(&mut instance, &[]);
    assert_approx_eq!(
        f32,
        tick_8_result.state_snapshot.unwrap().neuron_states[3].voltage,
        0.5 * 0.8
    );

    instance.tick_no_input_until(1000);

    // both outgoing synapses of nid 0 are still depressed
    tick(&mut instance, &[0]);
    let tick_1001_result = tick_extract_snapshot(&mut instance, &[]);

    let expected_stp_factor = 0.8 * (1.0 - 0.6 * (-1000.0 / 800.0f32).exp());

    assert_approx_eq!(
        f32,
        tick_1001_result.state_snapshot.unwrap().neuron_states[2].voltage,
        0.5 * expected_stp_factor
    );

    instance.tick_no_input_until(1006);

    let tick_1006_result = tick_extract_snapshot(&mut instance, &[]);

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

    let mut tick_input = TickInput::new();

    for _ in 0..t_stop {
        tick_input.reset();

        tick_input.spiking_in_channel_ids = all_in_channels
            .choose_multiple(&mut rng, 5)
            .copied()
            .collect();

        tick_input.reward = reward_dist.sample(&mut rng);

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

    let tick_1_result = tick_extract_snapshot(&mut instance, &[]);

    assert_equal(tick_1_result.spiking_nids, [5, 6]);
    assert_equal(tick_1_result.spiking_out_channel_ids, [0, 1]);

    let state_snapshot = tick_1_result.state_snapshot.unwrap();
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

    // currently the only way to extract synapse info is via state snapshot in the tick result
    let tick_result = tick_extract_snapshot(&mut instance, &[]);
    let state_snapshot = tick_result.state_snapshot.unwrap();

    for synapse_state in state_snapshot.synapse_states {
        assert_ne!(synapse_state.pre_syn_nid, synapse_state.post_syn_nid);
    }
}
