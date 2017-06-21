local AbsNumCriterion, parent = torch.class('nn.AbsNumCriterion', 'nn.Criterion')

function AbsNumCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = true
   end
end

function AbsNumCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.AbsNumCriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function AbsNumCriterion:updateGradInput(input, target)
   input.THNN.AbsNumCriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   return self.gradInput
end