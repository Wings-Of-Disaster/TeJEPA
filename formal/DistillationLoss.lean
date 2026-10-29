import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Tactic.Linarith

/- Finite real-arithmetic properties of the CURRENT joint-CF TeReg and the
new weighted Patch / transformed Gram objective. No accuracy or convergence
claim; the spectral algorithm and floating-point implementation are not proved.
Angles are the actual random projections omega.S, omega.T and eta.T. -/
open scoped BigOperators InnerProductSpace
noncomputable section
namespace DistillationLoss

def phase (x : ℝ) : ℂ := Complex.exp ((x : ℂ)*Complex.I)

def joint {I K : Type*} [Fintype I] [Fintype K]
    (weights : K → ℝ) (s t anchor : I → K → ℝ) : ℝ :=
  ∑ k, weights k * Complex.normSq
    ((∑ i, phase (anchor i k)*(phase (s i k)-phase (t i k)))/(Fintype.card I : ℂ))

theorem joint_nonnegative {I K : Type*} [Fintype I] [Fintype K]
    (w : K → ℝ) (s t anchor : I → K → ℝ) (hw : ∀ k, 0 ≤ w k) :
    0 ≤ joint w s t anchor := by
  exact Finset.sum_nonneg fun k _ => mul_nonneg (hw k) (Complex.normSq_nonneg _)

theorem joint_teacher_zero {I K : Type*} [Fintype I] [Fintype K]
    (w : K → ℝ) (t anchor : I → K → ℝ) : joint w t t anchor = 0 := by
  simp [joint]

def transformedGram {I : Type*} [Fintype I] (s t : I → ℝ) : ℝ :=
  ∑ i, (s i-t i)^2

theorem transformedGram_nonnegative {I : Type*} [Fintype I] (s t : I → ℝ) :
    0 ≤ transformedGram s t :=
  Finset.sum_nonneg fun i _ => sq_nonneg (s i-t i)

theorem transformedGram_teacher_zero {I : Type*} [Fintype I] (t : I → ℝ) :
    transformedGram t t = 0 := by simp [transformedGram]

def reliablePatch {I : Type*} [Fintype I] (w error : I → ℝ) (fallback : ℝ) : ℝ :=
  if 0 < ∑ i, w i then (∑ i, w i*error i)/(∑ i, w i) else fallback

theorem reliablePatch_nonnegative {I : Type*} [Fintype I]
    (w error : I → ℝ) (fallback : ℝ) (hw : ∀ i, 0 ≤ w i)
    (he : ∀ i, 0 ≤ error i) (hf : 0 ≤ fallback) :
    0 ≤ reliablePatch w error fallback := by
  unfold reliablePatch
  split_ifs with h
  · exact div_nonneg (Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (he i)) h.le
  · exact hf

theorem reliablePatch_teacher_zero {I : Type*} [Fintype I] (w : I → ℝ) :
    reliablePatch w (fun _ => 0) 0 = 0 := by simp [reliablePatch]

theorem teacher_reliability_weight_bounds (floor temperature error : ℝ)
    (hf : 0 < floor) (hf1 : floor ≤ 1) (ht : 0 < temperature) (he : 0 ≤ error) :
    0 < floor+(1-floor)*Real.exp (-(error/temperature)) ∧
      floor+(1-floor)*Real.exp (-(error/temperature)) ≤ 1 := by
  have hp := Real.exp_pos (-(error/temperature))
  have hu : Real.exp (-(error/temperature)) ≤ 1 :=
    Real.exp_le_one_iff.mpr (neg_nonpos.mpr (div_nonneg he ht.le))
  have hlo := mul_nonneg (sub_nonneg.mpr hf1) hp.le
  have hhi := mul_le_mul_of_nonneg_left hu (sub_nonneg.mpr hf1)
  constructor <;> linarith

/- First-order gradient statement only. gR already includes its loss weight.
Does not cover AdamW, finite learning rates, prediction gradients, or accuracy. -/
theorem regularizer_budget_preserves_kd_descent {E : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (gKD gR : E) (rho : ℝ) (hKD : gKD ≠ 0)
    (hR : ‖gR‖ ≤ rho*‖gKD‖) (hr : rho < 1) :
    0 < ⟪gKD,gKD+gR⟫_ℝ := by
  have hn : 0 < ‖gKD‖ := norm_pos_iff.mpr hKD
  have hc := abs_real_inner_le_norm gKD gR
  have hl := neg_abs_le (⟪gKD,gR⟫_ℝ : ℝ)
  have hb := mul_le_mul_of_nonneg_left hR (norm_nonneg gKD)
  have hpos := mul_pos (sub_pos.mpr hr) (sq_pos_of_pos hn)
  rw [inner_add_right,real_inner_self_eq_norm_sq]
  nlinarith

#print axioms joint_nonnegative
#print axioms joint_teacher_zero
#print axioms transformedGram_nonnegative
#print axioms transformedGram_teacher_zero
#print axioms reliablePatch_nonnegative
#print axioms reliablePatch_teacher_zero
#print axioms teacher_reliability_weight_bounds
#print axioms regularizer_budget_preserves_kd_descent
end DistillationLoss
