use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Pre-built game tree stored as struct-of-arrays for cache efficiency.
/// Immutable after construction.
struct GameTree {
    is_terminal: Vec<bool>,
    term_is_fold: Vec<bool>,
    term_winner: Vec<usize>,
    term_pot_total: Vec<f64>,
    term_contrib: Vec<[f64; 2]>,
    node_player: Vec<usize>,
    node_num_actions: Vec<usize>,
    children_offset: Vec<usize>,
    children_flat: Vec<usize>,
    node_infoset_idx: Vec<usize>,
}

impl GameTree {
    fn children(&self, node_idx: usize) -> &[usize] {
        let start = self.children_offset[node_idx];
        let na = self.node_num_actions[node_idx];
        &self.children_flat[start..start + na]
    }
}

struct InfoSetData {
    num_hands: usize,
    num_actions: usize,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
    strategy_cache: Option<Vec<f64>>,
    last_dcfr_iter: usize,
}

impl InfoSetData {
    fn new(num_hands: usize, num_actions: usize) -> Self {
        InfoSetData {
            num_hands,
            num_actions,
            regret_sum: vec![0.0; num_hands * num_actions],
            strategy_sum: vec![0.0; num_hands * num_actions],
            strategy_cache: None,
            last_dcfr_iter: 0,
        }
    }

    fn current_strategy(&mut self) -> &[f64] {
        if self.strategy_cache.is_some() {
            return self.strategy_cache.as_ref().unwrap();
        }
        let na = self.num_actions;
        let nh = self.num_hands;
        let uniform = 1.0 / na as f64;
        let mut strat = vec![0.0; nh * na];
        for h in 0..nh {
            let base = h * na;
            let mut normalizing = 0.0f64;
            for a in 0..na {
                let r = self.regret_sum[base + a];
                if r > 0.0 {
                    normalizing += r;
                }
            }
            if normalizing > 0.0 {
                for a in 0..na {
                    let r = self.regret_sum[base + a];
                    strat[base + a] = if r > 0.0 { r / normalizing } else { 0.0 };
                }
            } else {
                for a in 0..na {
                    strat[base + a] = uniform;
                }
            }
        }
        self.strategy_cache = Some(strat);
        self.strategy_cache.as_ref().unwrap()
    }

    fn mark_dirty(&mut self) {
        self.strategy_cache = None;
    }

    fn apply_dcfr_discount(&mut self, iteration: usize, alpha: f64, beta: f64, gamma: f64) {
        if self.last_dcfr_iter == iteration {
            return;
        }
        let n = self.num_hands * self.num_actions;
        for t in (self.last_dcfr_iter + 1)..=(iteration) {
            let tf = t as f64;
            let pos_base = tf.powf(alpha);
            let neg_base = tf.powf(beta);
            let pos_scale = pos_base / (pos_base + 1.0);
            let neg_scale = neg_base / (neg_base + 1.0);
            let strat_scale = (tf / (tf + 1.0)).powf(gamma);
            for i in 0..n {
                let r = self.regret_sum[i];
                if r > 0.0 {
                    self.regret_sum[i] = r * pos_scale;
                } else if r < 0.0 {
                    self.regret_sum[i] = r * neg_scale;
                }
            }
            for i in 0..n {
                self.strategy_sum[i] *= strat_scale;
            }
        }
        self.last_dcfr_iter = iteration;
        self.mark_dirty();
    }
}

struct ShowdownData {
    blocked_indices: [Vec<Vec<usize>>; 2],
    opp_sorted_indices: [Vec<usize>; 2],
    hand_strength_range: [Vec<(usize, usize)>; 2],
    opp_sorted_pos: [Vec<usize>; 2],
}

#[pyclass]
struct RustCFRTrainer {
    tree: GameTree,
    showdown: ShowdownData,
    infosets: [Vec<InfoSetData>; 2],
    pending_regret: [Vec<Vec<f64>>; 2],
    num_hands: [usize; 2],
    hand_weights: [Vec<f64>; 2],
    use_plus: bool,
    linear_weighting: bool,
    alternating: bool,
    use_dcfr: bool,
    dcfr_alpha: f64,
    dcfr_beta: f64,
    dcfr_gamma: f64,
    iteration: usize,
}

#[pymethods]
impl RustCFRTrainer {
    #[new]
    #[pyo3(signature = (game_data, config_data))]
    fn new(game_data: &Bound<'_, PyDict>, config_data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let use_plus: bool = config_data.get_item("use_plus")?.unwrap().extract()?;
        let linear_weighting: bool = config_data.get_item("linear_weighting")?.unwrap().extract()?;
        let alternating: bool = config_data.get_item("alternating")?.unwrap().extract()?;
        let use_dcfr: bool = config_data.get_item("use_dcfr")?.unwrap().extract()?;
        let dcfr_alpha: f64 = config_data.get_item("dcfr_alpha")?.unwrap().extract()?;
        let dcfr_beta: f64 = config_data.get_item("dcfr_beta")?.unwrap().extract()?;
        let dcfr_gamma: f64 = config_data.get_item("dcfr_gamma")?.unwrap().extract()?;

        let num_hands: Vec<usize> = game_data.get_item("num_hands")?.unwrap().extract()?;
        let hw0: Vec<f64> = game_data.get_item("hand_weights_0")?.unwrap().extract()?;
        let hw1: Vec<f64> = game_data.get_item("hand_weights_1")?.unwrap().extract()?;
        let bl0: Vec<Vec<usize>> = game_data.get_item("blocked_0")?.unwrap().extract()?;
        let bl1: Vec<Vec<usize>> = game_data.get_item("blocked_1")?.unwrap().extract()?;
        let os0: Vec<usize> = game_data.get_item("opp_sorted_0")?.unwrap().extract()?;
        let os1: Vec<usize> = game_data.get_item("opp_sorted_1")?.unwrap().extract()?;
        let hsr0: Vec<(usize, usize)> = game_data.get_item("hand_strength_range_0")?.unwrap().extract()?;
        let hsr1: Vec<(usize, usize)> = game_data.get_item("hand_strength_range_1")?.unwrap().extract()?;
        let osp0: Vec<usize> = game_data.get_item("opp_sorted_pos_0")?.unwrap().extract()?;
        let osp1: Vec<usize> = game_data.get_item("opp_sorted_pos_1")?.unwrap().extract()?;

        let node_types: Vec<i64> = game_data.get_item("node_types")?.unwrap().extract()?;
        let node_pot_totals: Vec<f64> = game_data.get_item("node_pot_totals")?.unwrap().extract()?;
        let node_contribs: Vec<(f64, f64)> = game_data.get_item("node_contribs")?.unwrap().extract()?;
        let node_tw: Vec<i64> = game_data.get_item("node_terminal_winners")?.unwrap().extract()?;
        let node_players: Vec<i64> = game_data.get_item("node_players")?.unwrap().extract()?;
        let node_children: Vec<Vec<usize>> = game_data.get_item("node_children")?.unwrap().extract()?;
        let node_isidx: Vec<i64> = game_data.get_item("node_infoset_idx")?.unwrap().extract()?;
        let num_is: Vec<usize> = game_data.get_item("num_infosets")?.unwrap().extract()?;
        let is_na: Vec<Vec<usize>> = game_data.get_item("infoset_num_actions")?.unwrap().extract()?;

        let n = node_types.len();
        let mut is_terminal_v = Vec::with_capacity(n);
        let mut term_is_fold_v = Vec::with_capacity(n);
        let mut term_winner_v = Vec::with_capacity(n);
        let mut term_pot_v = Vec::with_capacity(n);
        let mut term_contrib_v = Vec::with_capacity(n);
        let mut node_player_v = Vec::with_capacity(n);
        let mut node_na_v = Vec::with_capacity(n);
        let mut ch_offset_v = Vec::with_capacity(n);
        let mut ch_flat_v = Vec::new();
        let mut node_isidx_v = Vec::with_capacity(n);

        for i in 0..n {
            let is_term = node_types[i] == 0;
            is_terminal_v.push(is_term);
            if is_term {
                let is_fold = node_tw[i] >= 0;
                term_is_fold_v.push(is_fold);
                term_winner_v.push(if is_fold { node_tw[i] as usize } else { 0 });
                term_pot_v.push(node_pot_totals[i]);
                term_contrib_v.push([node_contribs[i].0, node_contribs[i].1]);
                node_player_v.push(0);
                node_na_v.push(0);
                ch_offset_v.push(ch_flat_v.len());
                node_isidx_v.push(0);
            } else {
                term_is_fold_v.push(false);
                term_winner_v.push(0);
                term_pot_v.push(0.0);
                term_contrib_v.push([0.0, 0.0]);
                node_player_v.push(node_players[i] as usize);
                let na = node_children[i].len();
                node_na_v.push(na);
                ch_offset_v.push(ch_flat_v.len());
                ch_flat_v.extend_from_slice(&node_children[i]);
                node_isidx_v.push(node_isidx[i] as usize);
            }
        }

        let tree = GameTree {
            is_terminal: is_terminal_v,
            term_is_fold: term_is_fold_v,
            term_winner: term_winner_v,
            term_pot_total: term_pot_v,
            term_contrib: term_contrib_v,
            node_player: node_player_v,
            node_num_actions: node_na_v,
            children_offset: ch_offset_v,
            children_flat: ch_flat_v,
            node_infoset_idx: node_isidx_v,
        };

        let showdown = ShowdownData {
            blocked_indices: [bl0, bl1],
            opp_sorted_indices: [os0, os1],
            hand_strength_range: [hsr0, hsr1],
            opp_sorted_pos: [osp0, osp1],
        };

        let mut is0 = Vec::with_capacity(num_is[0]);
        for &na in &is_na[0] { is0.push(InfoSetData::new(num_hands[0], na)); }
        let mut is1 = Vec::with_capacity(num_is[1]);
        for &na in &is_na[1] { is1.push(InfoSetData::new(num_hands[1], na)); }

        let mut pr0 = Vec::with_capacity(num_is[0]);
        for d in &is0 { pr0.push(vec![0.0; d.num_hands * d.num_actions]); }
        let mut pr1 = Vec::with_capacity(num_is[1]);
        for d in &is1 { pr1.push(vec![0.0; d.num_hands * d.num_actions]); }

        Ok(RustCFRTrainer {
            tree, showdown,
            infosets: [is0, is1],
            pending_regret: [pr0, pr1],
            num_hands: [num_hands[0], num_hands[1]],
            hand_weights: [hw0, hw1],
            use_plus, linear_weighting, alternating,
            use_dcfr, dcfr_alpha, dcfr_beta, dcfr_gamma,
            iteration: 0,
        })
    }

    fn run(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.iteration += 1;
            if self.alternating {
                for player in 0..2 {
                    for pr in self.pending_regret[player].iter_mut() {
                        for v in pr.iter_mut() { *v = 0.0; }
                    }
                    let rp = self.hand_weights[player].clone();
                    let ro = self.hand_weights[1 - player].clone();
                    let mut vals = vec![0.0; self.num_hands[player]];
                    self.traverse(0, player, &rp, &ro, &mut vals);
                    self.apply_regret_updates(player);
                }
            } else {
                for player in 0..2 {
                    for pr in self.pending_regret[player].iter_mut() {
                        for v in pr.iter_mut() { *v = 0.0; }
                    }
                }
                for player in 0..2 {
                    let rp = self.hand_weights[player].clone();
                    let ro = self.hand_weights[1 - player].clone();
                    let mut vals = vec![0.0; self.num_hands[player]];
                    self.traverse(0, player, &rp, &ro, &mut vals);
                }
                for player in 0..2 { self.apply_regret_updates(player); }
            }
        }
    }

    fn average_strategy_profile(&self, py: Python<'_>, infoset_keys: Vec<Vec<String>>, action_tokens: Vec<Vec<Vec<String>>>) -> PyResult<PyObject> {
        let outer = PyDict::new(py);
        for player in 0..2usize {
            let inner = PyDict::new(py);
            for (idx, infoset) in self.infosets[player].iter().enumerate() {
                let na = infoset.num_actions;
                let nh = infoset.num_hands;
                let uniform = 1.0 / na as f64;
                let matrix = PyList::empty(py);
                for h in 0..nh {
                    let base = h * na;
                    let mut total = 0.0f64;
                    for a in 0..na { total += infoset.strategy_sum[base + a]; }
                    let row = PyList::empty(py);
                    if total > 0.0 {
                        for a in 0..na { row.append(infoset.strategy_sum[base + a] / total)?; }
                    } else {
                        for _a in 0..na { row.append(uniform)?; }
                    }
                    matrix.append(row)?;
                }
                let key = &infoset_keys[player][idx];
                let tokens = &action_tokens[player][idx];
                let tl = PyList::new(py, tokens.iter())?;
                let pair = pyo3::types::PyTuple::new(py, &[tl.as_any(), matrix.as_any()])?;
                inner.set_item(key, pair)?;
            }
            outer.set_item(player as i64, inner)?;
        }
        Ok(outer.into())
    }

    fn get_iteration(&self) -> usize { self.iteration }

    fn exploitability(&self, base_pot: f64) -> (f64, f64) {
        let avg: [Vec<Vec<f64>>; 2] = [
            self.build_avg_strategies(0),
            self.build_avg_strategies(1),
        ];
        let vw: [Vec<f64>; 2] = [
            self.compute_valid_weights(0),
            self.compute_valid_weights(1),
        ];
        let mut br_total = 0.0f64;
        for tp in 0..2 {
            let opp = 1 - tp;
            let or_ = self.hand_weights[opp].clone();
            let mut vals = vec![0.0; self.num_hands[tp]];
            self.best_response_traverse(0, tp, &or_, &avg[opp], &mut vals);
            let nh = self.num_hands[tp];
            for h in 0..nh {
                if vw[tp][h] > 0.0 { vals[h] /= vw[tp][h]; } else { vals[h] = 0.0; }
            }
            let mut tot = 0.0f64;
            let mut tw = 0.0f64;
            for h in 0..nh {
                let j = self.hand_weights[tp][h] * vw[tp][h];
                tot += j * vals[h];
                tw += j;
            }
            if tw > 0.0 { br_total += tot / tw; }
        }
        ((br_total - base_pot) / 2.0, base_pot)
    }
}

impl RustCFRTrainer {
    fn traverse(&mut self, ni: usize, up: usize, rp: &[f64], ro: &[f64], vals: &mut [f64]) {
        if self.tree.is_terminal[ni] {
            self.terminal_values_idx(ni, up, ro, vals);
            return;
        }
        let player = self.tree.node_player[ni];
        let na = self.tree.node_num_actions[ni];
        let isidx = self.tree.node_infoset_idx[ni];
        let nh_up = self.num_hands[up];
        let nh_p = self.num_hands[player];
        let children: Vec<usize> = self.tree.children(ni).to_vec();

        if player != up {
            let strat = self.infosets[player][isidx].current_strategy().to_vec();
            for v in vals.iter_mut() { *v = 0.0; }
            let mut nro = vec![0.0; nh_p];
            let mut cv = vec![0.0; nh_up];
            for ai in 0..na {
                for h in 0..nh_p { nro[h] = ro[h] * strat[h * na + ai]; }
                for v in cv.iter_mut() { *v = 0.0; }
                self.traverse(children[ai], up, rp, &nro, &mut cv);
                for h in 0..nh_up { vals[h] += cv[h]; }
            }
            return;
        }

        if self.use_dcfr {
            self.infosets[player][isidx].apply_dcfr_discount(
                self.iteration, self.dcfr_alpha, self.dcfr_beta, self.dcfr_gamma);
        }

        let strat = self.infosets[player][isidx].current_strategy().to_vec();
        let mut av = vec![0.0; na * nh_p];
        let mut nrp = vec![0.0; nh_p];

        for ai in 0..na {
            for h in 0..nh_p { nrp[h] = rp[h] * strat[h * na + ai]; }
            let sl = &mut av[ai * nh_p..(ai + 1) * nh_p];
            self.traverse(children[ai], up, &nrp, ro, sl);
        }

        for h in 0..nh_p {
            let mut v = 0.0f64;
            for ai in 0..na { v += strat[h * na + ai] * av[ai * nh_p + h]; }
            vals[h] = v;
        }

        let rw = if self.linear_weighting && !self.use_plus && !self.use_dcfr {
            self.iteration as f64
        } else { 1.0 };

        if self.use_plus {
            let pend = &mut self.pending_regret[player][isidx];
            for h in 0..nh_p {
                let nv = vals[h];
                let base = h * na;
                for ai in 0..na {
                    pend[base + ai] += (av[ai * nh_p + h] - nv) * rw;
                }
            }
        } else {
            let rs = &mut self.infosets[player][isidx].regret_sum;
            for h in 0..nh_p {
                let nv = vals[h];
                let base = h * na;
                for ai in 0..na {
                    rs[base + ai] += (av[ai * nh_p + h] - nv) * rw;
                }
            }
            self.infosets[player][isidx].mark_dirty();
        }

        let ws = if self.linear_weighting && !self.use_dcfr { self.iteration as f64 } else { 1.0 };
        let ss = &mut self.infosets[player][isidx].strategy_sum;
        for h in 0..nh_p {
            let w = rp[h] * ws;
            if w == 0.0 { continue; }
            let base = h * na;
            for ai in 0..na { ss[base + ai] += w * strat[h * na + ai]; }
        }
    }

    #[inline]
    fn terminal_values_idx(&self, ni: usize, up: usize, ow: &[f64], vals: &mut [f64]) {
        if self.tree.term_is_fold[ni] {
            let winner = self.tree.term_winner[ni];
            let pt = self.tree.term_pot_total[ni];
            let c = self.tree.term_contrib[ni][up];
            let v = if winner == up { pt - c } else { -c };
            self.fold_values(up, v, ow, vals);
        } else {
            let pt = self.tree.term_pot_total[ni];
            let c = self.tree.term_contrib[ni][up];
            self.showdown_values(up, ow, pt, c, vals);
        }
    }

    fn fold_values(&self, player: usize, value: f64, ow: &[f64], vals: &mut [f64]) {
        let total: f64 = ow.iter().sum();
        if total <= 0.0 {
            for v in vals.iter_mut() { *v = 0.0; }
            return;
        }
        let blocked = &self.showdown.blocked_indices[player];
        for (h, bl) in blocked.iter().enumerate() {
            let mut bw = 0.0f64;
            for &idx in bl { bw += unsafe { *ow.get_unchecked(idx) }; }
            vals[h] = value * (total - bw);
        }
    }

    fn showdown_values(&self, player: usize, ow: &[f64], pt: f64, cp: f64, vals: &mut [f64]) {
        let tow: f64 = ow.iter().sum();
        if tow <= 0.0 {
            for v in vals.iter_mut() { *v = 0.0; }
            return;
        }
        let si = &self.showdown.opp_sorted_indices[player];
        let ns = si.len();
        let osp = &self.showdown.opp_sorted_pos[player];
        let mut prefix = vec![0.0f64; ns + 1];
        for i in 0..ns { prefix[i + 1] = prefix[i] + ow[si[i]]; }
        let blocked = &self.showdown.blocked_indices[player];
        let sr = &self.showdown.hand_strength_range[player];
        let nh = self.num_hands[player];
        let hp = pt / 2.0;
        for h in 0..nh {
            let (s, e) = sr[h];
            let mut ww = prefix[s];
            let mut tw = prefix[e] - prefix[s];
            let mut lw = tow - ww - tw;
            for &idx in &blocked[h] {
                let w = ow[idx];
                if w == 0.0 { continue; }
                let pos = osp[idx];
                if pos < s { ww -= w; }
                else if pos < e { tw -= w; }
                else { lw -= w; }
            }
            vals[h] = ww * pt + tw * hp - cp * (ww + tw + lw);
        }
    }

    fn build_avg_strategies(&self, player: usize) -> Vec<Vec<f64>> {
        self.infosets[player].iter().map(|is| {
            let na = is.num_actions;
            let nh = is.num_hands;
            let u = 1.0 / na as f64;
            let mut avg = vec![0.0; nh * na];
            for h in 0..nh {
                let b = h * na;
                let mut t = 0.0f64;
                for a in 0..na { t += is.strategy_sum[b + a]; }
                if t > 0.0 {
                    for a in 0..na { avg[b + a] = is.strategy_sum[b + a] / t; }
                } else {
                    for a in 0..na { avg[b + a] = u; }
                }
            }
            avg
        }).collect()
    }

    fn compute_valid_weights(&self, player: usize) -> Vec<f64> {
        let opp = 1 - player;
        let total: f64 = self.hand_weights[opp].iter().sum();
        if total <= 0.0 { return vec![0.0; self.num_hands[player]]; }
        self.showdown.blocked_indices[player].iter().map(|bl| {
            let mut bw = 0.0f64;
            for &idx in bl { bw += self.hand_weights[opp][idx]; }
            total - bw
        }).collect()
    }

    fn best_response_traverse(&self, ni: usize, tp: usize, or_: &[f64], oas: &[Vec<f64>], vals: &mut [f64]) {
        if self.tree.is_terminal[ni] {
            self.terminal_values_idx(ni, tp, or_, vals);
            return;
        }
        let player = self.tree.node_player[ni];
        let na = self.tree.node_num_actions[ni];
        let children = self.tree.children(ni);
        let isidx = self.tree.node_infoset_idx[ni];
        let nh_t = self.num_hands[tp];
        let nh_p = self.num_hands[player];

        if player != tp {
            let strat = &oas[isidx];
            for v in vals.iter_mut() { *v = 0.0; }
            let mut nro = vec![0.0; nh_p];
            let mut cv = vec![0.0; nh_t];
            for ai in 0..na {
                for h in 0..nh_p { nro[h] = or_[h] * strat[h * na + ai]; }
                for v in cv.iter_mut() { *v = 0.0; }
                self.best_response_traverse(children[ai], tp, &nro, oas, &mut cv);
                for h in 0..nh_t { vals[h] += cv[h]; }
            }
            return;
        }

        let mut av = vec![0.0; na * nh_t];
        for ai in 0..na {
            let sl = &mut av[ai * nh_t..(ai + 1) * nh_t];
            self.best_response_traverse(children[ai], tp, or_, oas, sl);
        }
        for h in 0..nh_t {
            let mut best = av[h];
            for ai in 1..na {
                let v = av[ai * nh_t + h];
                if v > best { best = v; }
            }
            vals[h] = best;
        }
    }

    fn apply_regret_updates(&mut self, player: usize) {
        if !self.use_plus { return; }
        let n = self.infosets[player].len();
        for idx in 0..n {
            let sz = self.infosets[player][idx].num_hands * self.infosets[player][idx].num_actions;
            for i in 0..sz {
                let nv = self.infosets[player][idx].regret_sum[i] + self.pending_regret[player][idx][i];
                self.infosets[player][idx].regret_sum[i] = if nv > 0.0 { nv } else { 0.0 };
            }
            self.infosets[player][idx].mark_dirty();
        }
    }
}

#[pymodule]
fn rust_cfr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCFRTrainer>()?;
    Ok(())
}
