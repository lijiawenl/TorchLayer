#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/AbsNumCriterion.cu"
#else

void THNN_(AbsNumCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 2, input, target);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);
  /*add by JiaWen Li*/

  /*accreal sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal)0, thrust::plus<accreal>(), abs_functor<real, accreal>());*/
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  accreal sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal)0, thrust::plus<accreal>(), absNum_functor<real, accreal>());

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
  if(sum >=0){
  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}else{
  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(-sum));
}
}

void THNN_(AbsNumCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<double, real>::to(sizeAverage ? 1./size : 1.);
  real norm1 = ScalarConvert<double, real>::to(sizeAverage ? -1./size : -1.);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);
  
  THCTensor_(resizeAs)(state, gradInput, input);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));/*fengzhuang chuanru de zhizhen*/
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));
  accreal sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal)0, thrust::plus<accreal>(), absNum_functor<real, accreal>());
 
  if(sum >=0){
  /*thrust::transform(input_data, input_data+size, target_data, gradInput_data, abs_updateGradInput_functor<real>(norm));*/
  thrust::fill(gradInput_data,gradInput_data+size,(real) norm);
  }else{
  thrust::fill(gradInput_data,gradInput_data+size,(real) norm1);
  }
  
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif

