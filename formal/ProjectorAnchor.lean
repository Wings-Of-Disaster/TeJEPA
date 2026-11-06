import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

open scoped BigOperators
noncomputable section
namespace ProjectorAnchor

def paired {I : Type*} [Fintype I] (s t : I → ℝ) : ℝ :=
  ∑ i, (s i-t i)^2

theorem paired_nonnegative {I : Type*} [Fintype I] (s t : I → ℝ) :
    0 ≤ paired s t := Finset.sum_nonneg fun i _ => sq_nonneg (s i-t i)

theorem paired_teacher_zero {I : Type*} [Fintype I] (t : I → ℝ) :
    paired t t = 0 := by simp [paired]

theorem constant_decomposition {I : Type*} [Fintype I]
    (t : I → ℝ) (c m : ℝ) (hm : ∑ i, (t i-m) = 0) :
    paired (fun _ => c) t =
      (∑ i, (t i-m)^2) + (Fintype.card I : ℝ)*(c-m)^2 := by
  have identity : ∀ i, (c-t i)^2 = (t i-m)^2+(c-m)^2-2*(c-m)*(t i-m) := by
    intro i
    ring
  unfold paired
  simp_rw [identity]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hm]
  simp [nsmul_eq_mul]

theorem constant_lower_bound {I : Type*} [Fintype I]
    (t : I → ℝ) (c m : ℝ) (hm : ∑ i, (t i-m) = 0) :
    (∑ i, (t i-m)^2) ≤ paired (fun _ => c) t := by
  rw [constant_decomposition t c m hm]
  exact le_add_of_nonneg_right (mul_nonneg (Nat.cast_nonneg _) (sq_nonneg _))

theorem constant_total_positive {I : Type*} [Fintype I]
    (t : I → ℝ) (c m weight rest : ℝ) (hm : ∑ i, (t i-m) = 0)
    (hv : 0 < ∑ i, (t i-m)^2) (hw : 0 < weight) (hr : 0 ≤ rest) :
    0 < rest + weight*paired (fun _ => c) t := by
  have hp : 0 < paired (fun _ => c) t :=
    lt_of_lt_of_le hv (constant_lower_bound t c m hm)
  exact add_pos_of_nonneg_of_pos hr (mul_pos hw hp)

#print axioms paired_nonnegative
#print axioms paired_teacher_zero
#print axioms constant_decomposition
#print axioms constant_lower_bound
#print axioms constant_total_positive
end ProjectorAnchor
