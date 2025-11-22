brainstorming limitations

- first look at limitations in hyperscalar and ibm-activation steering 
- we still get our planning trajectories from prompts so if they don't work them out method might not
- we go against RLHF directions but still meet resistance

from hypersteer docs/paper/hypersteer_paper.md


    7 Limitations
    -------------

    ##### Data

    A key limitation of our approach is the limited scope and quantity of the concept datasets. Using data with concepts of much greater complexity and difficulty from a model steering perspective would likely improve model performance and help make evaluation more robust. We also note that quality and robustness of concepts is bounded by the GemmaScope feature labels used to derive them, and collecting data from humans or other high quality sources is a feasible alternative. This is a key research priority we emphasize for future work.

    ##### Steering Sites

    All experiments in our work are limited to intervening on the residual stream activations of the base LM. There are other potentially more performant sites for intervention, including various points of the decoder block and during the attention computation. We also adopt the convention of prior work to intervene at all token positions; exploring more targeted interventions could reduce detrimental off-target steering effects and improve the overall steering score.

    ##### Compute

    Compared to supervised dictionary learning, the compute requirements of training a hypernetwork are large, as the number of trainable parameters significantly exceeds a ReFT-r1.

    ##### Model Scale

    Due to to compute constraints we only experimented with Gemma-2-2B architectures, which are worse instruction followers and in-context learners than the leading open source models with many more parameters. Training on models at a variety of scale would help cement HyperSteer ’s strong steering performance against the improved in-context learning ability of larger LMs.

    ##### Open Source Models

    Our approach requires white-box access to a model’s internals in order to use steering vectors, a limitation prompting does not encounter. Hence, we rely on the existence of sufficiently capable open source models as a basis for our research.