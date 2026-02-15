use pyo3::prelude::*;

use crate::abstraction;
use crate::card;
use crate::eval;
use crate::state;

/// Python module for poker-engine.
#[pymodule]
fn poker_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Card functions
    m.add_function(wrap_pyfunction!(py_card, m)?)?;
    m.add_function(wrap_pyfunction!(py_rank, m)?)?;
    m.add_function(wrap_pyfunction!(py_suit, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_card, m)?)?;
    m.add_function(wrap_pyfunction!(py_card_to_string, m)?)?;

    // Hand evaluation
    m.add_function(wrap_pyfunction!(py_eval5, m)?)?;
    m.add_function(wrap_pyfunction!(py_eval7, m)?)?;
    m.add_function(wrap_pyfunction!(py_eval5_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_eval7_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_compare_hands, m)?)?;
    m.add_function(wrap_pyfunction!(py_hand_category, m)?)?;

    // Game state
    m.add_class::<PyGameState>()?;
    m.add_class::<PyAction>()?;

    // Action abstraction
    m.add_class::<PyActionAbstraction>()?;

    // Card abstraction
    m.add_function(wrap_pyfunction!(py_preflop_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ehs, m)?)?;
    m.add_function(wrap_pyfunction!(py_ehs_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_earth_movers_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_kmeans_emd, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_river_buckets, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_turn_buckets, m)?)?;
    m.add_class::<PyCardAbstraction>()?;

    // Batch operations
    m.add_function(wrap_pyfunction!(py_batch_apply_action, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_terminal_utility, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_infoset_keys, m)?)?;

    // Constants
    m.add("NUM_CARDS", card::NUM_CARDS)?;
    m.add("NUM_RANKS", card::NUM_RANKS)?;
    m.add("NUM_SUITS", card::NUM_SUITS)?;
    m.add("SMALL_BLIND", state::SMALL_BLIND)?;
    m.add("BIG_BLIND", state::BIG_BLIND)?;
    m.add("NUM_PREFLOP_BUCKETS", abstraction::NUM_PREFLOP_BUCKETS)?;

    Ok(())
}

// --- Card bindings ---

#[pyfunction]
fn py_card(rank: u8, suit: u8) -> u8 {
    card::card(rank, suit)
}

#[pyfunction]
fn py_rank(card_val: u8) -> u8 {
    card::rank(card_val)
}

#[pyfunction]
fn py_suit(card_val: u8) -> u8 {
    card::suit(card_val)
}

#[pyfunction]
fn py_parse_card(s: &str) -> Option<u8> {
    card::parse_card(s)
}

#[pyfunction]
fn py_card_to_string(c: u8) -> String {
    card::card_to_string(c)
}

// --- Eval bindings ---

#[pyfunction]
fn py_eval5(cards: Vec<u8>) -> PyResult<u16> {
    if cards.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eval5 requires exactly 5 cards",
        ));
    }
    let arr: [u8; 5] = cards.try_into().unwrap();
    Ok(eval::eval5(&arr))
}

#[pyfunction]
fn py_eval7(cards: Vec<u8>) -> PyResult<u16> {
    if cards.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eval7 requires exactly 7 cards",
        ));
    }
    let arr: [u8; 7] = cards.try_into().unwrap();
    Ok(eval::eval7(&arr))
}

#[pyfunction]
fn py_eval5_batch(hands: Vec<Vec<u8>>) -> PyResult<Vec<u16>> {
    let mut results = Vec::with_capacity(hands.len());
    for h in &hands {
        if h.len() != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each hand must have exactly 5 cards",
            ));
        }
        let arr: [u8; 5] = h.as_slice().try_into().unwrap();
        results.push(eval::eval5(&arr));
    }
    Ok(results)
}

#[pyfunction]
fn py_eval7_batch(hands: Vec<Vec<u8>>) -> PyResult<Vec<u16>> {
    let mut results = Vec::with_capacity(hands.len());
    for h in &hands {
        if h.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each hand must have exactly 7 cards",
            ));
        }
        let arr: [u8; 7] = h.as_slice().try_into().unwrap();
        results.push(eval::eval7(&arr));
    }
    Ok(results)
}

#[pyfunction]
fn py_compare_hands(hand_a: Vec<u8>, hand_b: Vec<u8>) -> PyResult<i8> {
    if hand_a.len() != 7 || hand_b.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Each hand must have exactly 7 cards",
        ));
    }
    let a: [u8; 7] = hand_a.try_into().unwrap();
    let b: [u8; 7] = hand_b.try_into().unwrap();
    Ok(eval::compare_hands(&a, &b))
}

#[pyfunction]
fn py_hand_category(rank: u16) -> &'static str {
    eval::hand_category(rank)
}

// --- Action wrapper ---

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyAction {
    pub(crate) inner: state::Action,
}

#[pymethods]
impl PyAction {
    #[staticmethod]
    fn fold() -> Self {
        PyAction {
            inner: state::Action::Fold,
        }
    }
    #[staticmethod]
    fn check() -> Self {
        PyAction {
            inner: state::Action::Check,
        }
    }
    #[staticmethod]
    fn call() -> Self {
        PyAction {
            inner: state::Action::Call,
        }
    }
    #[staticmethod]
    fn raise_to(amount: u32) -> Self {
        PyAction {
            inner: state::Action::Raise(amount),
        }
    }
    #[staticmethod]
    fn all_in() -> Self {
        PyAction {
            inner: state::Action::AllIn,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__(&self, other: &PyAction) -> bool {
        self.inner == other.inner
    }

    #[getter]
    fn action_type(&self) -> &'static str {
        match self.inner {
            state::Action::Fold => "fold",
            state::Action::Check => "check",
            state::Action::Call => "call",
            state::Action::Raise(_) => "raise",
            state::Action::AllIn => "all_in",
        }
    }

    #[getter]
    fn amount(&self) -> Option<u32> {
        match self.inner {
            state::Action::Raise(a) => Some(a),
            _ => None,
        }
    }
}

// --- Game state wrapper ---

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyGameState {
    pub(crate) inner: state::GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (stack=200))]
    fn new(stack: u32) -> Self {
        PyGameState {
            inner: state::GameState::new(stack),
        }
    }

    fn deal_hole_cards(&mut self, p0_card0: u8, p0_card1: u8, p1_card0: u8, p1_card1: u8) {
        self.inner
            .deal_hole_cards([p0_card0, p0_card1], [p1_card0, p1_card1]);
    }

    fn deal_flop(&mut self, c0: u8, c1: u8, c2: u8) {
        self.inner.deal_flop([c0, c1, c2]);
    }

    fn deal_turn(&mut self, card: u8) {
        self.inner.deal_turn(card);
    }

    fn deal_river(&mut self, card: u8) {
        self.inner.deal_river(card);
    }

    fn legal_actions(&self) -> Vec<PyAction> {
        self.inner
            .legal_actions()
            .into_iter()
            .map(|a| PyAction { inner: a })
            .collect()
    }

    fn apply_action(&self, action: &PyAction) -> PyGameState {
        PyGameState {
            inner: self.inner.apply_action(action.inner),
        }
    }

    fn terminal_utility(&self, player: usize) -> f64 {
        self.inner.terminal_utility(player)
    }

    fn infoset_key(&self, player: usize) -> String {
        self.inner.infoset_key(player)
    }

    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal
    }

    #[getter]
    fn active_player(&self) -> u8 {
        self.inner.active_player
    }

    #[getter]
    fn street(&self) -> &'static str {
        match self.inner.street {
            state::Street::Preflop => "preflop",
            state::Street::Flop => "flop",
            state::Street::Turn => "turn",
            state::Street::River => "river",
        }
    }

    #[getter]
    fn pot(&self) -> u32 {
        self.inner.pot()
    }

    #[getter]
    fn stacks(&self) -> (u32, u32) {
        (self.inner.stacks[0], self.inner.stacks[1])
    }

    #[getter]
    fn pot_contrib(&self) -> (u32, u32) {
        (self.inner.pot_contrib[0], self.inner.pot_contrib[1])
    }

    #[getter]
    fn street_contrib(&self) -> (u32, u32) {
        (self.inner.street_contrib[0], self.inner.street_contrib[1])
    }

    #[getter]
    fn current_bet(&self) -> u32 {
        self.inner.current_bet
    }

    #[getter]
    fn last_raise_size(&self) -> u32 {
        self.inner.last_raise_size
    }

    #[getter]
    fn hole_cards(&self) -> ((u8, u8), (u8, u8)) {
        (
            (self.inner.hole_cards[0], self.inner.hole_cards[1]),
            (self.inner.hole_cards[2], self.inner.hole_cards[3]),
        )
    }

    #[getter]
    fn board(&self) -> Vec<u8> {
        self.inner.board[..self.inner.board_len as usize].to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "GameState(street={:?}, pot={}, active={}, terminal={})",
            self.inner.street,
            self.inner.pot(),
            self.inner.active_player,
            self.inner.is_terminal
        )
    }
}

// --- Action abstraction wrapper ---

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyActionAbstraction {
    inner: state::ActionAbstraction,
}

#[pymethods]
impl PyActionAbstraction {
    #[new]
    #[pyo3(signature = (bet_fractions=None))]
    fn new(bet_fractions: Option<Vec<f64>>) -> Self {
        let inner = match bet_fractions {
            Some(fracs) => state::ActionAbstraction::new(fracs),
            None => state::ActionAbstraction::default(),
        };
        PyActionAbstraction { inner }
    }

    fn abstract_actions(&self, state: &PyGameState) -> Vec<PyAction> {
        self.inner
            .abstract_actions(&state.inner)
            .into_iter()
            .map(|a| PyAction { inner: a })
            .collect()
    }

    fn translate_action(&self, state: &PyGameState, action: &PyAction) -> PyAction {
        PyAction {
            inner: self.inner.translate_action(&state.inner, action.inner),
        }
    }

    #[getter]
    fn bet_fractions(&self) -> Vec<f64> {
        self.inner.bet_fractions.clone()
    }

    fn __repr__(&self) -> String {
        format!("ActionAbstraction(bet_fractions={:?})", self.inner.bet_fractions)
    }
}

// --- Card abstraction bindings ---

#[pyfunction]
fn py_preflop_bucket(card0: u8, card1: u8) -> u16 {
    abstraction::preflop_bucket(card0, card1)
}

#[pyfunction]
fn py_compute_ehs(hero: Vec<u8>, board: Vec<u8>) -> PyResult<f64> {
    if hero.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hero must have exactly 2 cards",
        ));
    }
    Ok(abstraction::compute_ehs([hero[0], hero[1]], &board))
}

#[pyfunction]
#[pyo3(signature = (hero, board, num_bins=10))]
fn py_ehs_histogram(hero: Vec<u8>, board: Vec<u8>, num_bins: usize) -> PyResult<Vec<f64>> {
    if hero.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hero must have exactly 2 cards",
        ));
    }
    Ok(abstraction::ehs_histogram(
        [hero[0], hero[1]],
        &board,
        num_bins,
    ))
}

#[pyfunction]
fn py_earth_movers_distance(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Histograms must have the same length",
        ));
    }
    Ok(abstraction::earth_movers_distance(&a, &b))
}

#[pyfunction]
#[pyo3(signature = (histograms, k, max_iters=50))]
fn py_kmeans_emd(histograms: Vec<Vec<f64>>, k: usize, max_iters: usize) -> Vec<u16> {
    abstraction::kmeans_emd(&histograms, k, max_iters)
}

#[pyfunction]
#[pyo3(signature = (board, num_buckets=200, num_ehs_bins=10))]
fn py_compute_river_buckets(
    board: Vec<u8>,
    num_buckets: usize,
    num_ehs_bins: usize,
) -> PyResult<(Vec<Vec<u8>>, Vec<u16>)> {
    if board.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Board must have exactly 5 cards",
        ));
    }
    let board_arr: [u8; 5] = board.try_into().unwrap();
    let (hands, assignments) =
        abstraction::compute_river_buckets(&board_arr, num_buckets, num_ehs_bins);
    let hands_vecs: Vec<Vec<u8>> = hands.iter().map(|h| h.to_vec()).collect();
    Ok((hands_vecs, assignments))
}

#[pyfunction]
#[pyo3(signature = (board, num_buckets=200, river_num_buckets=200, river_ehs_bins=10))]
fn py_compute_turn_buckets(
    board: Vec<u8>,
    num_buckets: usize,
    river_num_buckets: usize,
    river_ehs_bins: usize,
) -> PyResult<(Vec<Vec<u8>>, Vec<u16>)> {
    if board.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Board must have exactly 4 cards for turn bucketing",
        ));
    }
    let board_arr: [u8; 4] = board.try_into().unwrap();
    let (hands, assignments) = abstraction::compute_turn_buckets(
        &board_arr,
        num_buckets,
        river_num_buckets,
        river_ehs_bins,
    );
    let hands_vecs: Vec<Vec<u8>> = hands.iter().map(|h| h.to_vec()).collect();
    Ok((hands_vecs, assignments))
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyCardAbstraction {
    inner: abstraction::CardAbstraction,
}

#[pymethods]
impl PyCardAbstraction {
    #[new]
    #[pyo3(signature = (preflop=169, flop=200, turn=200, river=200))]
    fn new(preflop: usize, flop: usize, turn: usize, river: usize) -> Self {
        PyCardAbstraction {
            inner: abstraction::CardAbstraction::new(preflop, flop, turn, river),
        }
    }

    fn num_buckets(&self, street: usize) -> PyResult<usize> {
        if street > 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Street must be 0-3",
            ));
        }
        Ok(self.inner.num_buckets(street))
    }

    #[getter]
    fn buckets_per_street(&self) -> Vec<usize> {
        self.inner.buckets_per_street.to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "CardAbstraction(buckets={:?})",
            self.inner.buckets_per_street
        )
    }
}

// --- Batch operations ---

#[pyfunction]
fn py_batch_apply_action(states: Vec<PyGameState>, action: &PyAction) -> Vec<PyGameState> {
    states
        .iter()
        .map(|s| PyGameState {
            inner: s.inner.apply_action(action.inner),
        })
        .collect()
}

#[pyfunction]
fn py_batch_terminal_utility(states: Vec<PyGameState>, player: usize) -> Vec<f64> {
    states
        .iter()
        .map(|s| s.inner.terminal_utility(player))
        .collect()
}

#[pyfunction]
fn py_batch_infoset_keys(states: Vec<PyGameState>, player: usize) -> Vec<String> {
    states
        .iter()
        .map(|s| s.inner.infoset_key(player))
        .collect()
}
