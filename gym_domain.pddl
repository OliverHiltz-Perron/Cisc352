(define (domain gym-planner)
    (:requirements :negative-preconditions :strips :typing)
    (:predicates (base-built) (goal-reached) (hypertrophy-built))
    (:action do-base-block
        :parameters ()
        :precondition (not (base-built))
        :effect (base-built)
    )
     (:action do-hypertrophy-block
        :parameters ()
        :precondition (and (base-built) (not (hypertrophy-built)))
        :effect (hypertrophy-built)
    )
     (:action do-peaking-block
        :parameters ()
        :precondition (and (hypertrophy-built) (not (goal-reached)))
        :effect (goal-reached)
    )
)