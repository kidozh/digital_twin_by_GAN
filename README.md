# Digital Twins by Generative Adversarial Nets

This project is aimed to simulate tool wear in digital way. We want to simulate signal in machining process so that we can do that in the following:

+ reinforcement learning in small data
+ visualization on deep learning model
+ simulate signal in machining process
+ transfer learning at few data

We use DCGAN model with tool wear label to achieve that. GAN training is fast and the validity of simulated result will be tested later.

## attention

Since sensor signal is totally different from vision, many adjustment for 1D signal and working condition is done. Be careful if you want to reuse this code to your environment.

# Demonstrate

We will post a beamer (localized in simplified Chinese) to demonstrate our idea, trick and work about this.

A homepage will be conducted after that.

# Result 

you can find result in directory `res`. It includes trained model and signal pictures generated for different tool wear stage. The signal is displayed in 7 dimension corresponding to sub-directory.

Since it's not as intuitive as image, it's hard for human to identify but maybe easy for ResNet model.

# Contact

Please feel free to contact with me in mail and issue way about academic consult. Also fork is welcomed. Other things involved with cooperation with certain region may be restricted under NPU's related regulation.

# LICENSE

MIT LICENSE