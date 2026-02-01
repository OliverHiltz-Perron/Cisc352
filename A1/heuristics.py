# =============================
# Student Names: Oliver Hiltz-Perron, Aidan Kooiman
# Group ID: 57
# Date: 2/1/2026
# =============================
# CISC 352
# heuristics.py
# desc:
#


#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete problem solution.

'''This file will contain different constraint propagators to be used within
   the propagators

1. ord_dh (worth 0.25/3 points)
    - a Variable ordering heuristic that chooses the next Variable to be assigned 
      according to the Degree heuristic

2. ord_mv (worth 0.25/3 points)
    - a Variable ordering heuristic that chooses the next Variable to be assigned 
      according to the Minimum-Remaining-Value heuristic


var_ordering == a function with the following template
    var_ordering(csp)
        ==> returns Variable

    csp is a CSP object---the heuristic can use this to get access to the
    Variables and constraints of the problem. The assigned Variables can be
    accessed via methods, the values assigned can also be accessed.

    var_ordering returns the next Variable to be assigned, as per the definition
    of the heuristic it implements.
   '''

def ord_dh(csp):
    ''' return next Variable to be assigned according to the Degree Heuristic '''
    # IMPLEMENT
    unassigned_vars = csp.get_all_unasgn_vars()
    
    # Check that there are unassigned variables
    if not unassigned_vars:
        return None
    
    # 
    max_var = None
    max_degree = -1
    
    # Iterate over every unassigned variable
    for var in unassigned_vars:
        # temp degree to compare with
        degree = 0
        # Iterate over all constraints with this variable
        for constraint in csp.get_cons_with_var(var):
            # Iterate over variables that are involved in with this constraint
            for scope_var in constraint.get_scope():
                # Check if this constraint has other unassigned variables
                if scope_var != var and not scope_var.is_assigned():
                    degree += 1
                    break
        
        if degree > max_degree:
            max_degree = degree
            max_var = var
    
    return max_var

def ord_mrv(csp):
    ''' return Variable to be assigned according to the Minimum Remaining Values heuristic '''
    # IMPLEMENT
    unassigned_vars = csp.get_all_unasgn_vars()
    
    # Check that there are unassigned variables
    if not unassigned_vars:
        return None
    
    # Set first entry to min to compare with
    min_var = unassigned_vars[0]
    min_domain_size = min_var.cur_domain_size()
    
    # Iterate over unassigned variables to compare to find smallest domain
    for var in unassigned_vars[1:]:
        domain_size = var.cur_domain_size()
        if domain_size < min_domain_size:
            min_domain_size = domain_size
            min_var = var
    
    return min_var