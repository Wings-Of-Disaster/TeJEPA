import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
Finite conditional-mean prediction and a compatible joint optimum.
The conditional mean is specified by zero residual sums inside each view cell.
Teacher sufficiency and realizability are explicit hypotheses, never axioms.
These scalar results apply coordinatewise to vector squared-error prediction.
No claim about every stationary point, optimizer convergence or accuracy.
-/
open scoped BigOperators
noncomputable section
namespace TeacherPrediction

def risk {I : Type*} [Fintype I] (p y : I → ℝ) : ℝ := ∑ i, (p i-y i)^2

theorem risk_decomposition {I : Type*} [Fintype I] (p m y : I → ℝ)
    (hcross : ∑ i, (p i-m i)*(m i-y i)=0) :
    risk p y = risk m y + ∑ i, (p i-m i)^2 := by
  have identity : ∀ i, (p i-y i)^2 =
      (m i-y i)^2+(p i-m i)^2+2*((p i-m i)*(m i-y i)) := by
    intro i
    ring
  unfold risk
  simp_rw [identity]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hcross]
  ring

theorem conditional_residual_orthogonal {I J : Type*} [Fintype I] [Fintype J]
    [DecidableEq J] (view : I → J) (q m : J → ℝ) (y : I → ℝ)
    (hmean : ∀ j, ∑ i ∈ Finset.univ with view i=j, (m j-y i)=0) :
    ∑ i, (q (view i)-m (view i))*(m (view i)-y i)=0 := by
  classical
  rw [← Finset.sum_fiberwise Finset.univ view
    (fun i => (q (view i)-m (view i))*(m (view i)-y i))]
  apply Finset.sum_eq_zero
  intro j _
  calc
    (∑ i ∈ Finset.univ with view i=j, (q (view i)-m (view i))*(m (view i)-y i)) =
        ∑ i ∈ Finset.univ with view i=j, (q j-m j)*(m j-y i) := by
          apply Finset.sum_congr rfl
          intro i hi
          rw [(Finset.mem_filter.mp hi).2]
    _ = (q j-m j)*(∑ i ∈ Finset.univ with view i=j, (m j-y i)) := by
      rw [Finset.mul_sum]
    _ = 0 := by rw [hmean j, mul_zero]

theorem conditional_mean_optimal {I J : Type*} [Fintype I] [Fintype J]
    [DecidableEq J] (view : I → J) (q m : J → ℝ) (y : I → ℝ)
    (hmean : ∀ j, ∑ i ∈ Finset.univ with view i=j, (m j-y i)=0) :
    risk (fun i => m (view i)) y ≤ risk (fun i => q (view i)) y := by
  rw [risk_decomposition _ _ _ (conditional_residual_orthogonal view q m y hmean)]
  exact le_add_of_nonneg_right (Finset.sum_nonneg fun i _ => sq_nonneg _)

theorem teacher_factorization {I A : Type*} (s t : I → A) (q : A → ℝ) (m : I → ℝ)
    (hmatch : ∀ i, s i=t i) (hsufficient : ∀ i, q (t i)=m i) :
    ∀ i, q (s i)=m i := by
  intro i
  rw [hmatch i, hsufficient i]

theorem teacher_factorized_risk {I A : Type*} [Fintype I]
    (s t : I → A) (q : A → ℝ) (m y : I → ℝ)
    (hmatch : ∀ i, s i=t i) (hsufficient : ∀ i, q (t i)=m i) :
    risk (fun i => q (s i)) y = risk m y := by
  unfold risk
  simp only [teacher_factorization s t q m hmatch hsufficient]

theorem constant_centering {I : Type*} [Fintype I]
    (m : I → ℝ) (c center : ℝ) (hcenter : ∑ i, (m i-center)=0) :
    (∑ i, (c-m i)^2) =
      (∑ i, (m i-center)^2)+(Fintype.card I : ℝ)*(c-center)^2 := by
  have identity : ∀ i, (c-m i)^2 =
      (m i-center)^2+(c-center)^2-2*(c-center)*(m i-center) := by
    intro i
    ring
  simp_rw [identity]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hcenter]
  simp [nsmul_eq_mul]

theorem constant_prediction_strictly_suboptimal {I : Type*} [Fintype I]
    (m y : I → ℝ) (c center : ℝ)
    (hcross : ∑ i, (c-m i)*(m i-y i)=0)
    (hcenter : ∑ i, (m i-center)=0)
    (hinformative : 0 < ∑ i, (m i-center)^2) :
    risk m y < risk (fun _ => c) y := by
  rw [risk_decomposition _ _ _ hcross, constant_centering m c center hcenter]
  have hn : 0 ≤ (Fintype.card I : ℝ)*(c-center)^2 :=
    mul_nonneg (Nat.cast_nonneg _) (sq_nonneg _)
  linarith

theorem weighted_joint_lower_bound (reg prediction baseline lam mu : ℝ)
    (hr : 0 ≤ reg) (hp : baseline ≤ prediction) (hlam : 0 < lam) (hmu : 0 < mu) :
    mu*baseline ≤ lam*reg+mu*prediction := by
  have h1 := mul_nonneg hlam.le hr
  have h2 := mul_le_mul_of_nonneg_left hp hmu.le
  linarith

theorem weighted_joint_optimal_iff (reg prediction baseline lam mu : ℝ)
    (hr : 0 ≤ reg) (hp : baseline ≤ prediction) (hlam : 0 < lam) (hmu : 0 < mu) :
    lam*reg+mu*prediction=mu*baseline ↔ reg=0 ∧ prediction=baseline := by
  constructor
  · intro h
    have h1 := mul_nonneg hlam.le hr
    have h2 := mul_nonneg hmu.le (sub_nonneg.mpr hp)
    have hreg : lam*reg=0 := by nlinarith
    have hpred : mu*(prediction-baseline)=0 := by nlinarith
    have hr0 := (mul_eq_zero.mp hreg).resolve_left (ne_of_gt hlam)
    have hp0 := (mul_eq_zero.mp hpred).resolve_left (ne_of_gt hmu)
    exact ⟨hr0,sub_eq_zero.mp hp0⟩
  · rintro ⟨hr0,hp0⟩
    rw [hr0,hp0]
    ring

#print axioms risk_decomposition
#print axioms conditional_residual_orthogonal
#print axioms conditional_mean_optimal
#print axioms teacher_factorization
#print axioms teacher_factorized_risk
#print axioms constant_centering
#print axioms constant_prediction_strictly_suboptimal
#print axioms weighted_joint_lower_bound
#print axioms weighted_joint_optimal_iff
end TeacherPrediction
