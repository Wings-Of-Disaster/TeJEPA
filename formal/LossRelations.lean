import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
Structural audit of the complete objective. This file deliberately does not
assert that every loss has an independent zero set: exact CLS matching implies
RKD matching. It proves the implication, its converse failure, and finite-error
non-equivalence; covered-patch and full-Gram observations are distinguished.
All witnesses are feature-level statements, not encoder optimization claims.
-/
open scoped BigOperators
noncomputable section
namespace LossRelations

def pointError {X : Type*} [Fintype X] (s t : X → ℝ) : ℝ := ∑ x, (s x-t x)^2

theorem pointError_zero_iff {X : Type*} [Fintype X] (s t : X → ℝ) :
    pointError s t=0 ↔ s=t := by
  constructor
  · intro h
    funext x
    have hz := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j (_ : j ∈ Finset.univ) => sq_nonneg (s j-t j))).mp h x (Finset.mem_univ x)
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hz)
  · intro h
    subst t
    simp [pointError]

def kl {I : Type*} [Fintype I] (p q : I → ℝ) : ℝ :=
  ∑ i, -q i*Real.log (p i/q i)

theorem kl_self {I : Type*} [Fintype I] (p : I → ℝ) : kl p p=0 := by
  simp [kl,Real.log_div_self]

def softmax {I : Type*} [Fintype I] (z : I → ℝ) (i : I) : ℝ :=
  Real.exp (z i)/(∑ j, Real.exp (z j))

def relationalKL {X I : Type*} [Fintype I]
    (score : (X → ℝ) → I → I → ℝ) (s t : X → ℝ) : ℝ :=
  ∑ i, kl (softmax (score s i)) (softmax (score t i))

theorem cls_zero_implies_rkd_zero {X I : Type*} [Fintype X] [Fintype I]
    (score : (X → ℝ) → I → I → ℝ) (s t : X → ℝ)
    (h : pointError s t=0) : relationalKL score s t=0 := by
  have heq := (pointError_zero_iff s t).mp h
  rw [heq]
  simp [relationalKL,kl_self]

theorem no_cls_zero_rkd_nonzero_witness {X I : Type*} [Fintype X] [Fintype I]
    (score : (X → ℝ) → I → I → ℝ) :
    ¬ ∃ s t, pointError s t=0 ∧ relationalKL score s t≠0 := by
  rintro ⟨s,t,hcls,hrkd⟩
  exact hrkd (cls_zero_implies_rkd_zero score s t hcls)

theorem kl_term_bound (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
    q-p ≤ -q*Real.log (p/q) := by
  have h := mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos (div_pos hp hq)) hq.le
  have he : q*(p/q-1)=p-q := by field_simp
  rw [he] at h
  linarith

theorem kl_term_strict (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hne : p≠q) :
    q-p < -q*Real.log (p/q) := by
  have hn : p/q≠1 := by
    intro h
    exact hne (by simpa using (div_eq_iff (ne_of_gt hq)).mp h)
  have h := mul_lt_mul_of_pos_left (Real.log_lt_sub_one_of_pos (div_pos hp hq) hn) hq
  have he : q*(p/q-1)=p-q := by field_simp
  rw [he] at h
  linarith

theorem kl_strictly_positive {I : Type*} [Fintype I] (p q : I → ℝ)
    (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i=1) (hqs : ∑ i, q i=1) (hne : ∃ i, p i≠q i) :
    0 < kl p q := by
  obtain ⟨i,hi⟩ := hne
  have hsum : (∑ j, (q j-p j)) < kl p q :=
    Finset.sum_lt_sum (fun j _ => kl_term_bound _ _ (hp j) (hq j))
      ⟨i,Finset.mem_univ i,kl_term_strict _ _ (hp i) (hq i) hi⟩
  simpa [Finset.sum_sub_distrib,hps,hqs] using hsum

theorem weighted_variance_identity {I : Type*} [Fintype I]
    (p u : I → ℝ) (m : ℝ) (hsum : ∑ i, p i=1) (hmean : ∑ i, p i*u i=m) :
    (∑ i, p i*(u i-m)^2)=(∑ i, p i*(u i)^2)-m^2 := by
  have identity : ∀ i, p i*(u i-m)^2 =
      p i*(u i)^2-2*m*(p i*u i)+m^2*p i := by
    intro i
    ring
  simp_rw [identity]
  rw [Finset.sum_add_distrib,Finset.sum_sub_distrib,
    ← Finset.mul_sum,← Finset.mul_sum,hmean,hsum]
  ring

theorem fisher_quadratic_nonnegative {I : Type*} [Fintype I]
    (p u : I → ℝ) (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i=1) :
    0 ≤ (∑ i, p i*(u i)^2)-(∑ i, p i*u i)^2 := by
  rw [← weighted_variance_identity p u (∑ i, p i*u i) hsum rfl]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (hp i) (sq_nonneg _)

abbrev Feature := (Bool × Bool) → ℝ

def teacher : Feature := fun x => if x.1=x.2 then 1 else 0
def scaled : Feature := fun x => 2*teacher x
def collapsed : Feature := fun x => if x.2 then 0 else 1

def dot2 (s : Feature) (i j : Bool) : ℝ := ∑ d, s (i,d)*s (j,d)
def cosine2 (s : Feature) (i j : Bool) : ℝ :=
  dot2 s i j/(Real.sqrt (dot2 s i i)*Real.sqrt (dot2 s j j))

theorem scaled_cls_error : pointError scaled teacher=2 := by
  norm_num [pointError,scaled,teacher,Fintype.sum_prod_type,Fintype.sum_bool]

theorem collapsed_cls_error : pointError collapsed teacher=2 := by
  norm_num [pointError,collapsed,teacher,Fintype.sum_prod_type,Fintype.sum_bool]

theorem scaled_cosines_match : ∀ i j, cosine2 scaled i j=cosine2 teacher i j := by
  intro i j
  cases i <;> cases j <;>
    norm_num [cosine2,dot2,scaled,teacher,Fintype.sum_bool]

theorem scaled_rkd_zero : relationalKL cosine2 scaled teacher=0 := by
  have heq : cosine2 scaled=cosine2 teacher := by
    funext i j
    exact scaled_cosines_match i j
  simp [relationalKL,heq,kl_self]

theorem rkd_zero_does_not_imply_cls_zero :
    relationalKL cosine2 scaled teacher=0 ∧ pointError scaled teacher≠0 := by
  exact ⟨scaled_rkd_zero,by rw [scaled_cls_error]; norm_num⟩

def rkdAt (tau : ℝ) (s t : Feature) : ℝ :=
  relationalKL (fun f i j => cosine2 f i j/tau) s t

theorem scaled_rkdAt_zero (tau : ℝ) : rkdAt tau scaled teacher=0 := by
  have heq : ∀ i, (fun j => cosine2 scaled i j/tau)=(fun j => cosine2 teacher i j/tau) := by
    intro i
    funext j
    rw [scaled_cosines_match]
  unfold rkdAt relationalKL
  simp_rw [heq]
  simp [kl_self]

def teacherProb (tau : ℝ) (i j : Bool) : ℝ :=
  if i=j then Real.exp (1/tau)/(Real.exp (1/tau)+1) else 1/(Real.exp (1/tau)+1)

theorem teacher_softmax (tau : ℝ) (i j : Bool) :
    softmax (fun j => cosine2 teacher i j/tau) j=teacherProb tau i j := by
  cases i <;> cases j <;>
    norm_num [softmax,cosine2,dot2,teacher,teacherProb,Fintype.sum_bool,add_comm]

theorem collapsed_softmax (tau : ℝ) (i j : Bool) :
    softmax (fun j => cosine2 collapsed i j/tau) j=1/2 := by
  have he : Real.exp (1/tau)≠0 := ne_of_gt (Real.exp_pos _)
  cases i <;> cases j <;>
    norm_num [softmax,cosine2,dot2,collapsed,Fintype.sum_bool] <;>
    field_simp <;> ring

theorem teacherProb_positive (tau : ℝ) (i j : Bool) : 0 < teacherProb tau i j := by
  have hd : 0 < Real.exp (1/tau)+1 := by linarith [Real.exp_pos (1/tau)]
  by_cases h : i=j
  · simpa [teacherProb,h] using div_pos (Real.exp_pos _) hd
  · simpa [teacherProb,h] using div_pos (by norm_num : (0 : ℝ)<1) hd

theorem teacherProb_sum (tau : ℝ) (i : Bool) : (∑ j, teacherProb tau i j)=1 := by
  have hd : Real.exp (1/tau)+1≠0 := ne_of_gt (by linarith [Real.exp_pos (1/tau)] : 0 < Real.exp (1/tau)+1)
  cases i <;> simp [teacherProb,Fintype.sum_bool] <;> field_simp <;> ring

theorem teacherProb_diagonal_gt_half (tau : ℝ) (htau : 0 < tau) (i : Bool) :
    (1 : ℝ)/2 < teacherProb tau i i := by
  have he : 1 < Real.exp (1/tau) := Real.one_lt_exp_iff.mpr (div_pos (by norm_num) htau)
  have hd : 0 < Real.exp (1/tau)+1 := by linarith
  simp only [teacherProb,ite_true]
  apply (div_lt_div_iff₀ (by norm_num : (0 : ℝ)<2) hd).2
  linarith

theorem uniform_teacher_kl_positive (tau : ℝ) (htau : 0 < tau) (i : Bool) :
    0 < kl (fun _ : Bool => 1/2) (teacherProb tau i) := by
  apply kl_strictly_positive
  · intro j; norm_num
  · exact teacherProb_positive tau i
  · norm_num [Fintype.sum_bool]
  · exact teacherProb_sum tau i
  · exact ⟨i,ne_of_lt (teacherProb_diagonal_gt_half tau htau i)⟩

theorem collapsed_rkd_positive (tau : ℝ) (htau : 0 < tau) :
    0 < rkdAt tau collapsed teacher := by
  have h1 : ∀ i, softmax (fun j => cosine2 collapsed i j/tau)=(fun _ : Bool => 1/2) := by
    intro i; funext j; exact collapsed_softmax tau i j
  have h2 : ∀ i, softmax (fun j => cosine2 teacher i j/tau)=teacherProb tau i := by
    intro i; funext j; exact teacher_softmax tau i j
  unfold rkdAt relationalKL
  simp_rw [h1,h2]
  exact Finset.sum_pos (fun i _ => uniform_teacher_kl_positive tau htau i) Finset.univ_nonempty

theorem rkd_not_a_function_of_cls_value (tau : ℝ) (htau : 0 < tau) :
    ¬ ∃ f : ℝ → ℝ, ∀ s : Feature,
      rkdAt tau s teacher=f (pointError s teacher) := by
  rintro ⟨f,hf⟩
  have ha := hf scaled
  have hb := hf collapsed
  rw [scaled_cls_error,scaled_rkdAt_zero] at ha
  rw [collapsed_cls_error] at hb
  have hz : rkdAt tau collapsed teacher=0 := hb.trans ha.symm
  linarith [collapsed_rkd_positive tau htau]

def coveredError (s t : Bool → ℝ) : ℝ := (s false-t false)^2
def gram1 (s : Bool → ℝ) (i j : Bool) : ℝ := s i*s j
def gramError (s t : Bool → ℝ) : ℝ := ∑ i, ∑ j, (gram1 s i j-gram1 t i j)^2
def patchesT : Bool → ℝ := fun _ => 1
def patchesUncovered : Bool → ℝ := fun i => if i then -1 else 1
def patchesRotated : Bool → ℝ := fun _ => -1

theorem covered_patch_zero_gram_positive :
    coveredError patchesUncovered patchesT=0 ∧ gramError patchesUncovered patchesT=8 := by
  norm_num [coveredError,gramError,gram1,patchesUncovered,patchesT,Fintype.sum_bool]

theorem gram_zero_patch_positive :
    gramError patchesRotated patchesT=0 ∧ coveredError patchesRotated patchesT=4 := by
  norm_num [coveredError,gramError,gram1,patchesRotated,patchesT,Fintype.sum_bool]

theorem full_patch_zero_implies_any_gram_zero {X Y : Type*} [Fintype X]
    (s t : X → ℝ) (phi : (X → ℝ) → Y) (loss : Y → Y → ℝ)
    (hdiag : ∀ y, loss y y=0) (h : pointError s t=0) : loss (phi s) (phi t)=0 := by
  rw [(pointError_zero_iff s t).mp h]
  exact hdiag _

def alignmentError (a : ℝ) (s t : Bool → ℝ) : ℝ := ∑ i, (a*s i-t i)^2
def rawT : Bool → ℝ := fun i => if i then 1 else -1
def rawScaled : Bool → ℝ := fun i => 2*rawT i

theorem learned_head_can_absorb_scale :
    alignmentError (1/2) rawScaled rawT=0 ∧ pointError rawScaled rawT=2 := by
  norm_num [alignmentError,pointError,rawScaled,rawT,Fintype.sum_bool]

theorem backbone_match_does_not_identify_head :
    pointError rawT rawT=0 ∧ alignmentError (-1) rawT rawT=8 := by
  norm_num [alignmentError,pointError,rawT,Fintype.sum_bool]

def pred3 (g1 g2 localView : ℝ) : ℝ :=
  (g1-(g1+g2)/2)^2+(g2-(g1+g2)/2)^2+(localView-(g1+g2)/2)^2

theorem constant_projector_zero (c : ℝ) : pred3 c c c=0 := by
  unfold pred3
  ring

theorem predictor_can_change_with_teacher_terms_fixed : pred3 1 1 0=1 ∧ pred3 0 0 0=0 := by
  norm_num [pred3]

theorem projector_elimination_minimum (rest weight c : ℝ) (hw : 0 ≤ weight) :
    rest+weight*pred3 c c c=rest ∧
    ∀ a b v, rest ≤ rest+weight*pred3 a b v := by
  constructor
  · rw [constant_projector_zero,mul_zero,add_zero]
  · intro a b v
    have hp : 0 ≤ pred3 a b v := by
      unfold pred3
      exact add_nonneg (add_nonneg (sq_nonneg _) (sq_nonneg _)) (sq_nonneg _)
    exact le_add_of_nonneg_right (mul_nonneg hw hp)

#print axioms pointError_zero_iff
#print axioms kl_self
#print axioms cls_zero_implies_rkd_zero
#print axioms no_cls_zero_rkd_nonzero_witness
#print axioms kl_term_bound
#print axioms kl_term_strict
#print axioms kl_strictly_positive
#print axioms weighted_variance_identity
#print axioms fisher_quadratic_nonnegative
#print axioms scaled_cls_error
#print axioms collapsed_cls_error
#print axioms scaled_cosines_match
#print axioms scaled_rkd_zero
#print axioms rkd_zero_does_not_imply_cls_zero
#print axioms scaled_rkdAt_zero
#print axioms teacher_softmax
#print axioms collapsed_softmax
#print axioms teacherProb_positive
#print axioms teacherProb_sum
#print axioms teacherProb_diagonal_gt_half
#print axioms uniform_teacher_kl_positive
#print axioms collapsed_rkd_positive
#print axioms rkd_not_a_function_of_cls_value
#print axioms covered_patch_zero_gram_positive
#print axioms gram_zero_patch_positive
#print axioms full_patch_zero_implies_any_gram_zero
#print axioms learned_head_can_absorb_scale
#print axioms backbone_match_does_not_identify_head
#print axioms constant_projector_zero
#print axioms predictor_can_change_with_teacher_terms_fixed
#print axioms projector_elimination_minimum
end LossRelations
