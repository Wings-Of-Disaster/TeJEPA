import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
An exact finite example distinguishing paired joint matching from marginal
matching. For two distinct teacher values, swap their student correspondences.
Let feature and anchor be the respective Gaussian off-diagonal kernel values.
The empirical joint-kernel V-statistic is computed explicitly below.

The ring lemmas also apply to matrix roots: inverse recovery and injectivity
are conditional on the supplied roots satisfying their square equations.
Construction, positive semidefiniteness and differentiation of matrix roots
are not silently assumed to have been formalized by these ring lemmas.
-/
noncomputable section
namespace JointAnchoring

def swappedJoint (feature anchor : ℝ) : ℝ :=
  ((2-2*feature)+(2-2*feature)+anchor*(2*feature-2)+anchor*(2*feature-2))/4

theorem swapped_joint_factorization (feature anchor : ℝ) :
    swappedJoint feature anchor = (1-feature)*(1-anchor) := by
  unfold swappedJoint
  ring

theorem swapped_joint_positive (feature anchor : ℝ)
    (hf : feature < 1) (ha : anchor < 1) : 0 < swappedJoint feature anchor := by
  rw [swapped_joint_factorization]
  exact mul_pos (sub_pos.mpr hf) (sub_pos.mpr ha)

theorem swapped_marginal_zero (feature : ℝ) : swappedJoint feature 1 = 0 := by
  rw [swapped_joint_factorization]
  ring

theorem shift_roots_injective {R : Type*} [Ring R]
    (g h a b c shift : R)
    (ha : a*a=g+shift) (hb : b*b=h+shift) (he : a-c=b-c) : g=h := by
  have hab : a=b := sub_left_inj.mp he
  have hh : g+shift=h+shift := by
    calc
      g+shift = a*a := ha.symm
      _ = b*b := by rw [hab]
      _ = h+shift := hb
  exact add_right_cancel hh

#print axioms swapped_joint_factorization
#print axioms swapped_joint_positive
#print axioms swapped_marginal_zero
#print axioms shift_roots_injective
end JointAnchoring
