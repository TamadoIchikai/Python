 def test_evaluation(rule_params, sim_config, label="Test"):
    """Test single cost evaluation with detailed output."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {label}")
    print("=" * 70)
    
    dkp_table, dkd_table = controller.decode_rules_from_params(rule_params)
    helper.print_rule_table(dkp_table, "dKp Rule Table")
    helper.print_rule_table(dkd_table, "dKd Rule Table")
    
    start = time.perf_counter()
    cost = optimizer.simulation_cost_FuzzyPID(rule_params, sim_config)
    elapsed = time.perf_counter() - start
    
    print(f"\nCost: {cost:.6f}")
    print(f"Evaluation time: {elapsed:.3f}s")
    print("=" * 70)
    
    return cost, elapsed
 # GBO parameters
    nP = 20
    MaxIt = 10000
    dim = 98
    lb = np.ones(dim, dtype=np.float64)
    ub = np.ones(dim, dtype=np.float64) * 7
    
    print("\n" + "=" * 70)
    print("GBO OPTIMIZATION SETTINGS")
    print("=" * 70)
    print(f"Population size: {nP}")
    print(f"Max iterations: {MaxIt}")
    print(f"Dimensions: {dim} (7x7 rule table)")
    print(f"Est. time per evaluation: {eval_time:.2f}s")
    print(f"Est. time per iteration: {eval_time * nP * 2:.1f}s")
    print(f"Est. total time: {eval_time * nP * 2 * MaxIt / 60:.1f} minutes")
    print(f"Cost weights: ISE={sim_config['w_ISE']}, ITAE={sim_config['w_ITAE']}, "
          f"IAE={sim_config['w_IAE']}, torque_U={sim_config['w_U']}")
    print("=" * 70)
    
    # Run GBO
    print("\n" + "=" * 70)
    print("STARTING GBO OPTIMIZATION...")
    print("=" * 70 + "\n")
    

Best_Cost, Best_Params, Convergence_curve = GBO(
        nP, MaxIt, lb, ub, dim,
        simulation_cost_FuzzyPID, sim_config,
        use_parallel=True,
        n_workers=min(nP, os.cpu_count())
    )
    
    total_time = time.perf_counter() - start_time
    
    # Results
    Best_dKp, Best_dKd = decode_rules_from_params(Best_Params)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best cost: {Best_Cost:.6f}")
    print(f"Improvement: {(default_cost - Best_Cost) / default_cost * 100:.2f}%")
    
    print_rule_table(Best_dKp, "OPTIMIZED dKp Rule Table")
    print_rule_table(Best_dKd, "OPTIMIZED dKd Rule Table")
    
    # Save results
    os.makedirs("FuzzyLogicOut", exist_ok=True)
    
    np.savez('FuzzyLogicOut/fuzzy_optimization_results.npz',
             Best_Cost=Best_Cost,
             Best_Params=Best_Params,
             Best_dKp=np.array(Best_dKp),
             Best_dKd=np.array(Best_dKd),
             Convergence=Convergence_curve,
             default_cost=default_cost)
    print("\nResults saved to FuzzyLogicOut/")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(Convergence_curve, 'b-', lw=2)
        ax[0].axhline(default_cost, color='r', ls='--', label=f'Default: {default_cost:.4f}')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Best Cost')
        ax[0].set_title('GBO Convergence')
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].semilogy(Convergence_curve, 'b-', lw=2)
        ax[1].axhline(default_cost, color='r', ls='--')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Best Cost (log)')
        ax[1].set_title('Convergence (Log Scale)')
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('FuzzyLogicOut/convergence.png', dpi=150)
    except Exception as e:
        print(f"Plot error: {e}")
    
    # Output code
    print("\n" + "=" * 70)
    print("COPY THIS TO YOUR CODE:")
    print("=" * 70)
    print(f"ruleTable_dKp_optimized = {Best_dKp}")
    print(f"\nruleTable_dKd_optimized = {Best_dKd}")
