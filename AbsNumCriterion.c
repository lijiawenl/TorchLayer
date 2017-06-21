#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/AbsNumCriterion.c"
#else

void THNN_(AbsNumCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage)
{
/*added by Jiawen Li*/
  real sum = 0;
  real sum1 = 0;
  real sum2 = 0;
  THNN_CHECK_NELEMENT(input, target);
  TH_TENSOR_APPLY2(real, input, real, target,
    sum1 += *input_data ;
    sum2 +=  *target_data;
    
  );
  /*after integer,then calculate the error*/
 sum = (sum1-sum2>=0?sum1-sum2:sum2-sum1);

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}


/*define function to */

void THNN_(AbsNumCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
 
  /*define the sum error of the crowd number*/
  real sum1 = 0;
  real sum2 = 0;
 
  TH_TENSOR_APPLY2(real, input, real, target,
    sum1 += *input_data ;
    sum2 +=  *target_data;
    
  );
  /*after integer,then calculate the error*/

  /*if (sizeAverage)
    sum /= THTensor_(nElement)(input);*/


  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);
  real norm1 = -1*(sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);

  if (sum1-sum2 >=0){
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data =  norm ;
  );
  }else{
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data =  norm1 ;
  );
  }
  
}

#endif
