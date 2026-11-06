import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-!
Exact real-arithmetic lemmas for a theoretical distillation objective.
This file proves the scalar spectral inverse/divided difference, injectivity
of the ACTUAL epsilon-stabilized normalization on normed real spaces, and
strictly-positive weighting/common-zero algebra. Matrix square-root calculus,
joint-CF population identification, numerical iteration error and optimization
are NOT asserted as axioms or claimed to be formally verified here.
-/
open scoped BigOperators
noncomputable section
namespace SpectralDistillation

def shiftedRoot (δ x : ℝ) : ℝ := Real.sqrt (x+δ)-Real.sqrt δ

theorem shiftedRoot_inverse (δ x : ℝ) (hδ : 0 ≤ δ) (hx : 0 ≤ x) :
    shiftedRoot δ x ^ 2 + 2*Real.sqrt δ*shiftedRoot δ x = x := by
  have ha := Real.sq_sqrt (add_nonneg hx hδ)
  have hb := Real.sq_sqrt hδ
  unfold shiftedRoot
  nlinarith

theorem shiftedRoot_injective (δ x y : ℝ) (hδ : 0 ≤ δ)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (h : shiftedRoot δ x = shiftedRoot δ y) : x=y := by
  have h1 := shiftedRoot_inverse δ x hδ hx
  have h2 := shiftedRoot_inverse δ y hδ hy
  rw [h] at h1
  linarith

theorem shiftedRoot_divided_difference (δ x y : ℝ) (hδ : 0 < δ)
    (hx : 0 ≤ x) (hy : 0 ≤ y) :
    shiftedRoot δ x-shiftedRoot δ y =
      (x-y)/(Real.sqrt (x+δ)+Real.sqrt (y+δ)) := by
  have ha := Real.sq_sqrt (add_nonneg hx hδ.le)
  have hb := Real.sq_sqrt (add_nonneg hy hδ.le)
  have hp : 0 < Real.sqrt (x+δ)+Real.sqrt (y+δ) :=
    add_pos_of_pos_of_nonneg (Real.sqrt_pos.mpr (by linarith)) (Real.sqrt_nonneg _)
  apply (eq_div_iff (ne_of_gt hp)).2
  unfold shiftedRoot
  nlinarith

theorem shiftedRoot_teacher_zero (δ : ℝ) : shiftedRoot δ 0 = 0 := by
  simp [shiftedRoot]

def stabilized {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (ε : ℝ) (x : E) : E := (‖x‖+ε)⁻¹ • x

theorem stabilized_injective {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (ε : ℝ) (hε : 0 < ε) : Function.Injective (stabilized (E:=E) ε) := by
  intro x y h
  have hx : 0 < ‖x‖+ε := add_pos_of_nonneg_of_pos (norm_nonneg _) hε
  have hy : 0 < ‖y‖+ε := add_pos_of_nonneg_of_pos (norm_nonneg _) hε
  have hn := congrArg norm h
  unfold stabilized at hn
  simp only [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hx),
    abs_of_pos (inv_pos.mpr hy)] at hn
  have hr : ‖x‖/(‖x‖+ε)=‖y‖/(‖y‖+ε) := by
    simpa only [div_eq_mul_inv, mul_comm] using hn
  have hc := (div_eq_div_iff (ne_of_gt hx) (ne_of_gt hy)).mp hr
  have hxy : ‖x‖=‖y‖ := by nlinarith
  have hs := congrArg (fun z : E => (‖x‖+ε) • z) h
  unfold stabilized at hs
  rw [hxy] at hs
  simpa only [smul_smul, mul_inv_cancel₀ (ne_of_gt hy), one_smul] using hs

theorem positive_weighted_zero_iff {I : Type*} [Fintype I]
    (w error : I → ℝ) (hw : ∀ i, 0 < w i) (he : ∀ i, 0 ≤ error i) :
    (∑ i, w i*error i)=0 ↔ ∀ i, error i=0 := by
  constructor
  · intro h i
    have hz := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j (_ : j ∈ Finset.univ) => mul_nonneg (hw j).le (he j))).mp h i (Finset.mem_univ i)
    exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt (hw i))
  · intro h
    simp [h]

theorem combined_zero_iff (reg patch gram lam mu : ℝ)
    (hr : 0 ≤ reg) (hp : 0 ≤ patch) (hg : 0 ≤ gram)
    (hlam : 0 < lam) (hmu : 0 < mu) :
    reg+lam*patch+mu*gram=0 ↔ reg=0 ∧ patch=0 ∧ gram=0 := by
  constructor
  · intro h
    have h1 : 0 ≤ lam*patch := mul_nonneg hlam.le hp
    have h2 : 0 ≤ mu*gram := mul_nonneg hmu.le hg
    have hr0 : reg=0 := by linarith
    have hp0 : lam*patch=0 := by linarith
    have hg0 : mu*gram=0 := by linarith
    exact ⟨hr0,(mul_eq_zero.mp hp0).resolve_left (ne_of_gt hlam),
      (mul_eq_zero.mp hg0).resolve_left (ne_of_gt hmu)⟩
  · rintro ⟨hr0,hp0,hg0⟩
    simp [hr0,hp0,hg0]

#print axioms shiftedRoot_inverse
#print axioms shiftedRoot_injective
#print axioms shiftedRoot_divided_difference
#print axioms shiftedRoot_teacher_zero
#print axioms stabilized_injective
#print axioms positive_weighted_zero_iff
#print axioms combined_zero_iff
end SpectralDistillation
