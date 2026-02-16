use criterion::{black_box, criterion_group, criterion_main, Criterion};
use poker_engine::card::{self, card};
use poker_engine::ev::{EvContext, NUM_HANDS};
use std::time::Duration;

/// Kuhn poker: 3 hands (J, Q, K) — each player gets one card.
fn bench_kuhn(c: &mut Criterion) {
    let ctx = EvContext::new(&[]);
    let mut reach = vec![0.0; NUM_HANDS];
    let mut count = 0;
    for i in 0..NUM_HANDS {
        if !ctx.board_blocked[i] && count < 3 {
            reach[i] = 1.0 / 3.0;
            count += 1;
        }
    }

    c.bench_function("cf_values_kuhn_3hands", |b| {
        b.iter(|| {
            black_box(ctx.compute_cf_values(&reach, &reach));
        })
    });
}

/// Leduc poker: 6 hands (3 ranks × 2 suits).
fn bench_leduc(c: &mut Criterion) {
    let ctx = EvContext::new(&[]);
    let mut reach = vec![0.0; NUM_HANDS];
    let mut count = 0;
    for i in 0..NUM_HANDS {
        if !ctx.board_blocked[i] && count < 6 {
            reach[i] = 1.0 / 6.0;
            count += 1;
        }
    }

    c.bench_function("cf_values_leduc_6hands", |b| {
        b.iter(|| {
            black_box(ctx.compute_cf_values(&reach, &reach));
        })
    });
}

/// River HUNL: ~1081 valid hands — NAIVE (no precomputed matrix).
fn bench_river_hunl_naive(c: &mut Criterion) {
    let board = [
        card(card::RANK_A, card::SUIT_SPADES),
        card(card::RANK_K, card::SUIT_HEARTS),
        card(card::RANK_Q, card::SUIT_DIAMONDS),
        card(card::RANK_J, card::SUIT_CLUBS),
        card(card::RANK_T, card::SUIT_SPADES),
    ];
    let ctx = EvContext::new(&board);
    let valid_count = ctx.board_blocked.iter().filter(|&&b| !b).count();

    let mut reach = vec![0.0; NUM_HANDS];
    for i in 0..NUM_HANDS {
        if !ctx.board_blocked[i] {
            reach[i] = 1.0 / valid_count as f64;
        }
    }

    c.bench_function(
        &format!("cf_values_river_naive_{}hands", valid_count),
        |b| {
            b.iter(|| {
                black_box(ctx.compute_cf_values(&reach, &reach));
            })
        },
    );
}

/// River HUNL: ~1081 valid hands — WITH precomputed payoff matrix.
fn bench_river_hunl_precomputed(c: &mut Criterion) {
    let board = [
        card(card::RANK_A, card::SUIT_SPADES),
        card(card::RANK_K, card::SUIT_HEARTS),
        card(card::RANK_Q, card::SUIT_DIAMONDS),
        card(card::RANK_J, card::SUIT_CLUBS),
        card(card::RANK_T, card::SUIT_SPADES),
    ];
    let mut ctx = EvContext::new(&board);
    ctx.precompute_payoffs();
    let valid_count = ctx.board_blocked.iter().filter(|&&b| !b).count();

    let mut reach = vec![0.0; NUM_HANDS];
    for i in 0..NUM_HANDS {
        if !ctx.board_blocked[i] {
            reach[i] = 1.0 / valid_count as f64;
        }
    }

    c.bench_function(
        &format!("cf_values_river_precomputed_{}hands", valid_count),
        |b| {
            b.iter(|| {
                black_box(ctx.compute_cf_values(&reach, &reach));
            })
        },
    );
}

/// Benchmark precompute_payoffs construction time.
fn bench_precompute_payoffs(c: &mut Criterion) {
    let board = [
        card(card::RANK_A, card::SUIT_SPADES),
        card(card::RANK_K, card::SUIT_HEARTS),
        card(card::RANK_Q, card::SUIT_DIAMONDS),
        card(card::RANK_J, card::SUIT_CLUBS),
        card(card::RANK_T, card::SUIT_SPADES),
    ];

    c.bench_function("precompute_payoffs_river", |b| {
        b.iter(|| {
            let mut ctx = EvContext::new(&board);
            ctx.precompute_payoffs();
            black_box(&ctx);
        })
    });
}

/// Full HUNL preflop: 1326 hands, no board.
fn bench_preflop_hunl(c: &mut Criterion) {
    let ctx = EvContext::new(&[]);

    let mut reach = vec![0.0; NUM_HANDS];
    for i in 0..NUM_HANDS {
        if !ctx.board_blocked[i] {
            reach[i] = 1.0 / NUM_HANDS as f64;
        }
    }

    c.bench_function("cf_values_preflop_1326hands", |b| {
        b.iter(|| {
            black_box(ctx.compute_cf_values(&reach, &reach));
        })
    });
}

/// Benchmark context construction time.
fn bench_context_construction(c: &mut Criterion) {
    let board = [
        card(card::RANK_A, card::SUIT_SPADES),
        card(card::RANK_K, card::SUIT_HEARTS),
        card(card::RANK_Q, card::SUIT_DIAMONDS),
        card(card::RANK_J, card::SUIT_CLUBS),
        card(card::RANK_T, card::SUIT_SPADES),
    ];

    c.bench_function("ev_context_construction_river", |b| {
        b.iter(|| {
            black_box(EvContext::new(&board));
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2));
    targets = bench_kuhn,
        bench_leduc,
        bench_river_hunl_naive,
        bench_river_hunl_precomputed,
        bench_precompute_payoffs,
        bench_preflop_hunl,
        bench_context_construction
}
criterion_main!(benches);
