ParametricZ.m
--main ParametricZ abstract class
Affine_ParametricZ.m
--concrete subclass of ParametricZ: most general, uses tensor toolbox
LowRank_Matrix_Recovery_ParametricZ.m
--concrete subclass of ParametricZ
LowRank_Plus_Sparse_Matrix_Recovery_ParametricZ.m
--concrete subclass of ParametricZ
Multiple_Snapshot_ParametricZ.m
--concrete subclass of ParametricZ, uses PLinTrans objects

PLinTrans.m
--abstract class to implement a parametric linear operator
Matrix_PLinTrans.m
--a concrete subclass of PLinTrans, used for explicit (dense or sparse) matrices
Calibration_PLinTrans.m
--a concrete subclass of PLinTrans, of form Diag(H*b)*A0

PBiGAMP.m
--main file to run PBiGAMP: needs problem, priors, options
PBiGAMPProblem.m
--problem class
PBiGAMPOpt.m
--options class
EMPBiGAMP.m
--main file to run EMPBiGAMP.m: wrapper around PBiGAMP.m
PBiGAMPsimple.m
--a wrapper for PBiGAMP.m that accepts a 3D array/tensor rather than a Problem

vec.m
--convenience function to vectorize a matrix
path_setup.m
--set useful paths

processResultsScripts/
--gathers data from results*/ and plots in figs/*/
resultsAffine/
resultsCalibration/
resultsLrps/
resultsTrivial/
--mat files for various problems
figs/
--contains plots from processResultsScripts

superComputer/
--perl scripts for running stuff on oakley

trial_LowRank_Matrix_Recovery.m
trial_LowRank_Plus_Sparse_Matrix_Recovery.m
trial_affine.m
trial_calibration.m
--these run a single monte-carlo trial
tfocs_linop.m
--interfaces with TFOCS in trial_LowRank_Plus_Sparse_Matrix_Recovery.m 

script_affine.m
script_calibration.m
script_low_rank_plus_sparse.m
--these call trial_* to build up a phase plane

verify_formulation_calibration.m
verify_formulation_low_rank_matrix_recovery.m
verify_formulation_low_rank_plus_sparse_matrix_recovery.m
verify_formulation_multiple_snapshot.m
verify_formulation_uniform_variance_low_rank_matrix_recovery.m
--these show examples of building and testing ParametricZ concrete subclasses


------

Recommended procedure for building new applications:

-Build the ParametricZ object
-Build a unit test to compare the results to results using the generic
tensor version. Follow the verify_formulation_* style
-Build a trial_* function that does a single test condition
-Build a script_* function to run over the phase planes
-Use the submit* scripts in the superComputer folder to efficiently run
on Oakley. 
-Build a process_* script to combine and display the phase plane results
