torchlogix.functional.GradFactor
================================

.. currentmodule:: torchlogix.functional

.. autoclass:: GradFactor

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~GradFactor.__init__
      ~GradFactor.apply
      ~GradFactor.backward
      ~GradFactor.forward
      ~GradFactor.jvp
      ~GradFactor.mark_dirty
      ~GradFactor.mark_non_differentiable
      ~GradFactor.mark_shared_storage
      ~GradFactor.maybe_clear_saved_tensors
      ~GradFactor.name
      ~GradFactor.register_hook
      ~GradFactor.register_prehook
      ~GradFactor.save_for_backward
      ~GradFactor.save_for_forward
      ~GradFactor.set_materialize_grads
      ~GradFactor.setup_context
      ~GradFactor.vjp
      ~GradFactor.vmap
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~GradFactor.dirty_tensors
      ~GradFactor.generate_vmap_rule
      ~GradFactor.materialize_grads
      ~GradFactor.metadata
      ~GradFactor.needs_input_grad
      ~GradFactor.next_functions
      ~GradFactor.non_differentiable
      ~GradFactor.requires_grad
      ~GradFactor.saved_for_forward
      ~GradFactor.saved_tensors
      ~GradFactor.saved_variables
      ~GradFactor.to_save
   
   