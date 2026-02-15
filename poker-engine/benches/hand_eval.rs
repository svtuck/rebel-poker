use criterion::{black_box, criterion_group, criterion_main, Criterion};
use poker_engine::card;
use poker_engine::eval;

fn bench_eval5(c: &mut Criterion) {
    // Pre-generate hands
    let hands: Vec<[u8; 5]> = (0..1000)
        .map(|i| {
            let cards: Vec<u8> = (0..52).collect();
            [
                cards[i % 52],
                cards[(i * 7 + 1) % 52],
                cards[(i * 13 + 2) % 52],
                cards[(i * 17 + 3) % 52],
                cards[(i * 23 + 4) % 52],
            ]
        })
        .collect();

    c.bench_function("eval5", |b| {
        b.iter(|| {
            for hand in &hands {
                black_box(eval::eval5(hand));
            }
        })
    });
}

fn bench_eval7(c: &mut Criterion) {
    let hands: Vec<[u8; 7]> = (0..1000)
        .map(|i| {
            [
                (i % 52) as u8,
                ((i * 7 + 1) % 52) as u8,
                ((i * 13 + 2) % 52) as u8,
                ((i * 17 + 3) % 52) as u8,
                ((i * 23 + 4) % 52) as u8,
                ((i * 29 + 5) % 52) as u8,
                ((i * 31 + 6) % 52) as u8,
            ]
        })
        .collect();

    c.bench_function("eval7", |b| {
        b.iter(|| {
            for hand in &hands {
                black_box(eval::eval7(hand));
            }
        })
    });
}

criterion_group!(benches, bench_eval5, bench_eval7);
criterion_main!(benches);
