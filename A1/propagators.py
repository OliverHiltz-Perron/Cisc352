# =============================
# Student Names: Oliver Hiltz-Perron, Aidan Kooiman
# Group ID: 57
# Date: February 1, 2026
# =============================
# CISC 352
# propagators.py
# desc:
#


#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete problem solution.

'''This file will contain different constraint propagators to be used within
   bt_search.

    1. prop_FC (worth 0.5/3 marks)
        - a propagator function that propagates according to the FC algorithm that 
          check constraints that have exactly one Variable in their scope that has 
          not assigned with a value, and prune appropriately

    2. prop_GAC (worth 0.5/3 marks)
        - a propagator function that propagates according to the GAC algorithm, as 
          covered in lecture

   propagator == a function with the following template
      propagator(csp, newly_instantiated_variable=None)
           ==> returns (True/False, [(Variable, Value), (Variable, Value) ...]

      csp is a CSP object---the propagator can use this to get access
      to the variables and constraints of the problem. The assigned Variables
      can be accessed via methods, the values assigned can also be accessed.

      newly_instaniated_variable is an optional argument.
      if newly_instantiated_variable is not None:
          then newly_instantiated_variable is the most
           recently assigned Variable of the search.
      else:
          progator is called before any assignments are made
          in which case it must decide what processing to do
           prior to any Variables being assigned. SEE BELOW

       The propagator returns True/False and a list of (Variable, Value) pairs.
       Return is False if a deadend has been detected by the propagator.
       in this case bt_search will backtrack
       return is true if we can continue.

      The list of Variable values pairs are all of the values
      the propagator pruned (using the Variable's prune_value method).
      bt_search NEEDS to know this in order to correctly restore these
      values when it undoes a Variable assignment.

      NOTE propagator SHOULD NOT prune a value that has already been
      pruned! Nor should it prune a value twice

      PROPAGATOR called with newly_instantiated_variable = None
      PROCESSING REQUIRED:
        for plain backtracking (where we only check fully instantiated
        constraints)
        we do nothing...return true, []

        for forward checking (where we only check constraints with one
        remaining Variable)
        we look for unary constraints of the csp (constraints whose scope
        contains only one Variable) and we forward_check these constraints.

        for gac we establish initial GAC by initializing the GAC queue
        with all constaints of the csp


      PROPAGATOR called with newly_instantiated_variable = a Variable V
      PROCESSING REQUIRED:
         for plain backtracking we check all constraints with V (see csp method
         get_cons_with_var) that are fully assigned.

         for forward checking we forward check all constraints with V
         that have one unassigned Variable left

         for gac we initialize the GAC queue with all constraints containing V.
   '''

def prop_BT(csp, newVar=None):
    '''Do plain backtracking propagation. That is, do no
    propagation at all. Just check fully instantiated constraints'''

    if not newVar:
        return True, []
    for c in csp.get_cons_with_var(newVar):
        if c.get_n_unasgn() == 0:
            vals = []
            vars = c.get_scope()
            for var in vars:
                vals.append(var.get_assigned_value())
            if not c.check_tuple(vals):
                return False, []
    return True, []

def prop_FC(csp, newVar=None):
    '''Do forward checking. That is check constraints with
       only one uninstantiated Variable. Remember to keep
       track of all pruned Variable,value pairs and return '''

    pruned = [] # empty list for pruned elements

    if newVar is None: # base case for unassigned start variable
        return True, pruned
    
    constraints = csp.get_cons_with_var(newVar) # gets the current constraints from newVar check
    for c in constraints: # iterated through constraints
        unasgn_vars = [
            v for v in c.get_scope() if not (v.is_assigned())
        ] # a list for all vars in the scope
        if len(unasgn_vars) == 0:
            continue
        if len(unasgn_vars) == 1:
            y = unasgn_vars[0]

            for val in y.cur_domain(): # iterates through each value in the domain of y
                values = []
                for var in c.get_scope(): # iterates through each var in the scope of c
                    if var is y:
                        values.append(val)
                    else:
                        values.append(var.get_assigned_value())
                if not c.check_tuple(values): # if the tuple violates the constraint, it gets pruned
                    y.prune_value(val)
                    pruned.append((y,val))

            if y.cur_domain_size() == 0: # if the domain of y is now empty, then forward checking failed
                return False, pruned

    return True, pruned

    #IMPLEMENT


def prop_GAC(csp, assignment=None, newVar=None):
    '''Do GAC propagation. If newVar is None we do initial GAC enforce
       processing all constraints. Otherwise we do GAC enforce with
       constraints containing newVar on GAC Queue'''
    
    GAC_pruned = [] # empty list for pruned elements

    if newVar is None: # adds all constraints in CSP to the queue
        GAC_queue = list(csp.get_all_cons())
    else:
        GAC_queue = list(csp.get_cons_with_var(newVar)) # gets the newest variable for propagation
    
    while GAC_queue:
        constraint = GAC_queue.pop(0) # first constraint of queue

        for var in constraint.get_scope(): # iterates through each variable in the scope of constraints
            for val in var.cur_domain():
                if not constraint.has_support(var, val): # pruned if there is no support
                    var.prune_value(val)
                    GAC_pruned.append((var, val))

                    if var.cur_domain_size() == 0: # empty domain and GAC fails
                        return False, GAC_pruned
                    
                    for cons in csp.get_cons_with_var(var): # add all constraints back except the current constraint
                        if cons != constraint and cons not in GAC_queue:
                            GAC_queue.append(cons)

    return True, GAC_pruned

    #IMPLEMENT
