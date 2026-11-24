from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.sparse import isspmatrix_csc
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station.pwr_train import PWRTrainSession
from coupling.pwr_sparse import SparseLocalModeratorPWRSession
from coupling.pwr_moderator import LocalModeratorPWRSession
from coupling.refinement import coupled_pwr_mesh


def test_dense_default_hook_preserves_exact_matrix_object():
    a=np.array([[1.,1e-300],[0.,-2.]])
    assert PWRTrainSession._solver_jacobian(None,a) is a


def test_sparse_hook_keeps_tiny_coefficients_and_is_owned_copy():
    obj=SparseLocalModeratorPWRSession.__new__(SparseLocalModeratorPWRSession)
    a=np.array([[1.,1e-300],[0.,-2.]])
    s=obj._solver_jacobian(a)
    assert isspmatrix_csc(s) and s.nnz==3
    np.testing.assert_array_equal(s.toarray(),a)
    a[0,0]=20.;assert s[0,0]==1.
    assert obj.last_sparse_jacobian['drop_tolerance'] is None


@pytest.mark.parametrize('matrix',[np.ones(3),np.ones((2,3)),np.array([[float('nan')]])])
def test_incomplete_jacobian_refused(matrix):
    obj=SparseLocalModeratorPWRSession.__new__(SparseLocalModeratorPWRSession)
    with pytest.raises(ValueError,match='Jacobian'):obj._solver_jacobian(matrix)


@pytest.fixture(scope='module')
def sparse_seed():
    return SparseLocalModeratorPWRSession(**coupled_pwr_mesh(2),max_step_s=.05,rtol=2e-9)


def test_sparse_and_dense_start_at_identical_physical_state(sparse_seed):
    dense=LocalModeratorPWRSession(**coupled_pwr_mesh(2),max_step_s=.05,rtol=2e-9)
    assert dense.snapshot()['coupled_state']==sparse_seed.snapshot()['coupled_state']
    assert sparse_seed.snapshot()['sparse_jacobian'] is None


def test_sparse_integration_has_physical_budgets_and_sparse_statistics(sparse_seed):
    from scripts.qualify_pwr_distributed_fuel import checks
    s=deepcopy(sparse_seed);position=s.controls['rod_target']
    row=s.advance(.1,{'rod_target':position+.01})
    assert all(checks(row).values())
    assert row['sparse_jacobian']['nnz']<row['sparse_jacobian']['total_entries']
    assert row['sparse_jacobian']['all_nonzero_coefficients_retained']
    assert row['solver_statistics']['nlu']>0


def test_json_restart_recomputes_reference_and_repeats_suffix(sparse_seed):
    s=deepcopy(sparse_seed);s.advance(.1,{'rod_target':s.controls['rod_target']+.01})
    saved=json.loads(json.dumps(s.checkpoint(),allow_nan=False));restored=SparseLocalModeratorPWRSession.restore(saved)
    assert restored.checkpoint()==saved
    assert s.advance(.1)==restored.advance(.1) and s.checkpoint()==restored.checkpoint()


def test_dense_and_sparse_checkpoint_policies_cannot_be_silently_exchanged(sparse_seed):
    saved=sparse_seed.checkpoint()
    with pytest.raises(ValueError,match='source/format'):LocalModeratorPWRSession.restore(saved)
    bad=deepcopy(unseal(saved));bad['source']['numerical_policy_sha256']='0'*64
    with pytest.raises(ValueError,match='source/format'):SparseLocalModeratorPWRSession.restore(seal(bad))


def test_sparse_invalid_command_rolls_back_whole_interval(sparse_seed):
    s=deepcopy(sparse_seed);before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,{'rod_target':2.})
    assert s.checkpoint()==before
