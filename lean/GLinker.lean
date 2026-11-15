/-
  Formal guarantees for the pure algorithms in glinker-rs (core Lean 4, no Mathlib).

  Rust ↔ Lean correspondence
  * `gliner::decode::greedy`        ↔ `greedy`, theorem `greedy_pairwise`
  * `gliner::decode::conflicts`     ↔ `overlap` (flat_ner = true, multi_label = false), `overlap_symm`
  * threshold filter (`p > t`)      ↔ `keepAbove`, theorem `keepAbove_mono`
  * `fuzz::lcs` / `fuzz::ratio`     ↔ `lcs`, theorems `lcs_le_left`, `lcs_le_right`, `lcs_comm`
                                      (so ratio = 2·LCS/(|a|+|b|) ∈ [0,1] and is symmetric)
  * `encode::span_tensors` mask     ↔ `spanValid`, theorem `spanValid_in_bounds`
  * `metrics::mcc_binary` numerator ↔ `mccNum`, theorem `mccNum_swap_classes`
  Scores are modelled as `Nat` (fixed-point); the Rust code compares `f32` with the same `>`.
-/

namespace GLinker

/-! ## Greedy span selection is conflict-free -/

/-- `clashes conf x kept` — does `x` conflict with any already-kept element? -/
def clashes {α : Type} (conf : α → α → Bool) (x : α) : List α → Bool
  | [] => false
  | k :: ks => conf k x || clashes conf x ks

/-- Every element is conflict-free with every element after it. -/
def AllPairsOk {α : Type} (conf : α → α → Bool) : List α → Prop
  | [] => True
  | k :: ks => clashes conf k ks = false ∧ AllPairsOk conf ks

/-- Greedy pass over spans (already sorted by descending score). Kept spans are consed. -/
def greedy {α : Type} (conf : α → α → Bool) : List α → List α → List α
  | kept, [] => kept
  | kept, x :: xs => if clashes conf x kept then greedy conf kept xs else greedy conf (x :: kept) xs

theorem greedy_pairwise {α : Type} (conf : α → α → Bool) :
    ∀ (xs kept : List α), AllPairsOk conf kept → AllPairsOk conf (greedy conf kept xs)
  | [], kept, h => h
  | x :: xs, kept, h => by
    unfold greedy
    split
    · exact greedy_pairwise conf xs kept h
    · next hc =>
      have hk : AllPairsOk conf (x :: kept) := by
        simp only [AllPairsOk]
        exact And.intro (by simpa using hc) h
      exact greedy_pairwise conf xs (x :: kept) hk

/-- Starting from nothing, greedy output is pairwise conflict-free. -/
theorem greedy_sound {α : Type} (conf : α → α → Bool) (xs : List α) :
    AllPairsOk conf (greedy conf [] xs) :=
  greedy_pairwise conf xs [] trivial

/-- Greedy never invents spans: everything kept came from the input. -/
theorem greedy_subset {α : Type} (conf : α → α → Bool) :
    ∀ (xs kept : List α) (y : α), y ∈ greedy conf kept xs → y ∈ kept ∨ y ∈ xs
  | [], kept, y, h => Or.inl h
  | x :: xs, kept, y, h => by
    unfold greedy at h
    split at h
    · cases greedy_subset conf xs kept y h with
      | inl hk => exact Or.inl hk
      | inr hx => exact Or.inr (List.Mem.tail _ hx)
    · cases greedy_subset conf xs (x :: kept) y h with
      | inl hk =>
        cases hk with
        | head => exact Or.inr (List.Mem.head _)
        | tail _ hk' => exact Or.inl hk'
      | inr hx => exact Or.inr (List.Mem.tail _ hx)

/-! ## The flat-NER conflict relation is symmetric -/

structure Span where
  s : Nat
  e : Nat
  deriving Repr

/-- GLiNER `has_overlapping` with `multi_label = false`: spans conflict unless disjoint. -/
def overlap (a b : Span) : Bool :=
  !(decide (b.e < a.s) || decide (a.e < b.s))

theorem overlap_symm (a b : Span) : overlap a b = overlap b a := by
  unfold overlap
  rw [Bool.or_comm]

/-! ## Threshold monotonicity: raising the threshold only removes predictions -/

def keepAbove (t : Nat) : List Nat → List Nat
  | [] => []
  | x :: xs => if t < x then x :: keepAbove t xs else keepAbove t xs

theorem keepAbove_mono {t1 t2 : Nat} (h : t1 ≤ t2) :
    ∀ (l : List Nat) (x : Nat), x ∈ keepAbove t2 l → x ∈ keepAbove t1 l
  | [], x, hx => by simp [keepAbove] at hx
  | y :: ys, x, hx => by
    unfold keepAbove at hx ⊢
    by_cases h2 : t2 < y
    · rw [if_pos h2] at hx
      rw [if_pos (Nat.lt_of_le_of_lt h h2)]
      cases hx with
      | head => exact List.Mem.head _
      | tail _ hm => exact List.Mem.tail _ (keepAbove_mono h ys x hm)
    · rw [if_neg h2] at hx
      have ih := keepAbove_mono h ys x hx
      by_cases h1 : t1 < y
      · rw [if_pos h1]; exact List.Mem.tail _ ih
      · rw [if_neg h1]; exact ih

/-! ## LCS (rapidfuzz `ratio` = 2·LCS/(|a|+|b|)) is bounded and symmetric -/

def mx (a b : Nat) : Nat := if a ≤ b then b else a

theorem mx_le {a b c : Nat} (h1 : a ≤ c) (h2 : b ≤ c) : mx a b ≤ c := by
  unfold mx; split <;> assumption

theorem mx_comm (a b : Nat) : mx a b = mx b a := by
  unfold mx; split <;> split <;> omega

def lcs : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | a :: as, b :: bs => if a = b then lcs as bs + 1 else mx (lcs as (b :: bs)) (lcs (a :: as) bs)

theorem lcs_le_left : ∀ (xs ys : List Nat), lcs xs ys ≤ xs.length
  | [], _ => by simp [lcs]
  | _ :: _, [] => by simp [lcs]
  | a :: as, b :: bs => by
    simp only [lcs]
    split
    · have := lcs_le_left as bs
      simp only [List.length_cons]; omega
    · apply mx_le
      · have := lcs_le_left as (b :: bs)
        simp only [List.length_cons]; omega
      · exact lcs_le_left (a :: as) bs

theorem lcs_comm : ∀ (xs ys : List Nat), lcs xs ys = lcs ys xs
  | [], [] => rfl
  | [], _ :: _ => by simp [lcs]
  | _ :: _, [] => by simp [lcs]
  | a :: as, b :: bs => by
    simp only [lcs]
    by_cases h : a = b
    · rw [if_pos h, if_pos h.symm, lcs_comm as bs]
    · have h' : ¬ b = a := fun e => h e.symm
      rw [if_neg h, if_neg h', lcs_comm as (b :: bs), lcs_comm (a :: as) bs, mx_comm]

theorem lcs_le_right (xs ys : List Nat) : lcs xs ys ≤ ys.length := by
  rw [lcs_comm]; exact lcs_le_left ys xs

/-- 2·LCS ≤ |a|+|b|, i.e. `ratio ≤ 1` (and `ratio ≥ 0` is trivial over Nat). -/
theorem ratio_le_one (xs ys : List Nat) : 2 * lcs xs ys ≤ xs.length + ys.length := by
  have := lcs_le_left xs ys
  have := lcs_le_right xs ys
  omega

/-! ## Span enumeration stays inside the text -/

/-- `span_tensors` marks (i, i+w) valid iff i + w < len. -/
def spanValid (len i w : Nat) : Bool := decide (i + w < len)

theorem spanValid_in_bounds (len i w : Nat) (h : spanValid len i w = true) :
    i < len ∧ i + w < len := by
  unfold spanValid at h
  have := of_decide_eq_true h
  omega

/-! ## MCC numerator is invariant under swapping the positive class -/

def mccNum (tp tn fp fn : Int) : Int := tp * tn - fp * fn

/-- Relabelling positives ↔ negatives maps (tp,tn,fp,fn) ↦ (tn,tp,fn,fp). -/
theorem mccNum_swap_classes (tp tn fp fn : Int) : mccNum tp tn fp fn = mccNum tn tp fn fp := by
  unfold mccNum
  rw [Int.mul_comm tp tn, Int.mul_comm fp fn]

/-- Perfect prediction (fp = fn = 0) gives a non-negative numerator. -/
theorem mccNum_perfect_nonneg (tp tn : Nat) : 0 ≤ mccNum tp tn 0 0 := by
  unfold mccNum
  rw [Int.mul_zero, Int.sub_zero]
  exact Int.mul_nonneg (by omega) (by omega)

end GLinker
