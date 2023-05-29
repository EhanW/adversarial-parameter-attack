### Experiment of adversarial parameters on VGG16

#### $L_0$ Attack
**'l_0.py'** is the code for $L_0$ attack show in the paper, it can be used directly.

 In the line 116, ```T``` is the value that how much proportion of  parameters can be changed. 
 The default value is 0.01, but you can change it as 0.0025-0.02 to see attack effect with different hyperparameter. 


#### $L_\infty$ Attack
**'l_inf.py'** is the code for $L_\infty$ attack show in the paper, it can be used directly.

 In the line 10, ```q``` is the value that how much can be changed for each parameter.  
The default value is 0.1, but you can change it as 0.01-0.1 to see the attack effect with different hyperparameter.

#### Pretrained Network
```vgg.pt``` a pretrained VGG-16 model. We used adversarial training to get it. 



