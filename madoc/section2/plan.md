## Section 2.1 - Optomiser comparison

Testing different optomisers. Since KANs tend to use fewer parameters than other NNs, applying different optomisers, especially those shown to improve performance on models with a small number of parameters when applied to PDE problems.

This paper claims the the LM optomisers delivers greatly superior performance on physics problems for models with a small number of paramters. This section aims to test that claim on the poisson equation in 2D, using the same tests and metrics as section1/section1_3.py.

This is informed by the paper - "Optimizing the optimizer for data driven deep neural networks and physics informed neural networks" - available at https://arxiv.org/abs/2205.07430

The four optomisers implemented in this paper are ADAM, LGBFS, LGBFS-L, and LM

## Section 2.2 - Adaptive density

Using the same heurisitc as is used in the pykan packages model.prune() function, instead of eliminating underperforming nodes, simply increase the density of the most important/sensitive nodes.

Do two tests. One with this as an alternative to the normal grid densification practice, and another also incorporating the regular grid densification, but with additional densification on imporant nodes.

Follow the instructions in heuristic.md to use the build in features of pykan, rather than making your own system

# 2.2.2 - Adaptive density (reduction)

In the interests of remaining below the interpolation threshold, reduce the density of nodes that would have been pruned down to a minimum. There are limitations here so far as.... hmmm.... ok.

Yeah ok so maybe once we approach the interpolation threshold.... we find some way to prune, reduce density etc on... some nodes? Like try to do what we can before we hit that threshold.

# Idea lol:
Would using a more dense MSE loss function actually prevent overfitting past the interpolation threshold?

More fine tuned grid size selection/changing, rather than picking from a list?

Maybe a one step look ahead to determine like the.... marge of each additional point haha.

## Section 2.3 - Merge_KAN

Similar to mixture of experts but very distinct. Train a variety of different KANs using different depths, basis functions (and maybe optomisers). Train each of these configurations from a variety of starting seeds. Then prune them until smaller KANs each reach the distinct sets of functional dependences mentioned in section 4 of the KANs paper in.

To merge kan, create a new KAN with one more layers. Then use the pykan api to make the output layer of each expert the top hidden layer of the merge KAN. That is if we had a 8 variable input {x_1....x_8} and say we found 3 different pruned KANs, with different dependencies. 

For example, if we have pruned cans with the following dependencies:
KAN_1 {x_1, x_2, x_6}
KAN_2 {x_3, x_4}
KAN_3 {x_2, x_5, x_7, x_8}

Eg if we have KANs of shape:

KAN_1: [3, 5, 1]
KAN_2: [2, 3, 1]
KAN_3: [4, 2, 1]

Then the merged KAN should be of shape [9, 10, 3, 1]

Then continue normal training for the Merge_KAN

Start with a fairly large number of KANs, then reduce the number at each of several stages, eventually combining into a single or handful of surviving KANS.

Eg you could start with 20 kans, then have 8 after the first merge, then 3 after the second.

Part of the idea behind merge KAN is inspired by the lack of catesrophic forgetting KANS have. I vibesways sus that this mean that different experts will "remember".... different sections/ properties? much more in this l8r I guess haha

I also think this and 2.4 could really benifit from being tested in a very high dimensional space? Like both cause curse of dimensionality and the weird complexity of these kinda spaces maybe that curse of forgetting shit will mega buff KAN probe or just KANS or even MERGE KAN just cause like there would be *so* many relationships and you would want some kind of way to *remember* them.

Is it possible to fold a KAN.... it can't be made into a single basis function?

Maybe try snapping all functions when merging?

Like KAN merge normal and KAN merge snap?

ALso maybe kan merge freeze for when the component KANS are put in, training on *them* is frozen.... results in a deeper models as there would need to be layers above but yeah should make sense right?

<!-- ## Section 2.3.1 -->


## Section 2.4 - KAN_Probe

Genetic evolution of KANs. Similar to Merge_KAN, but greatly extended. 

KAN_probe is best explained by allegory. Imagine a black hole is discovered to a physical reality with a much greater number of spatial dimensions. Each of these probes must explore this space. Probes enter this reality at different points, then figure out which direction to move (optomiser) - from each starting point, they should split into a variety of probes first differentiated simply by their optomisers. Each probe should continue for a specified number of epochs. After this, probes should report back to their mother probe and report their progress (Dense MSE, H1_norm, semi_norm etc). Based on which have progressed most, probes should prune their models, then recombined traits - such as super experts, as per merge_KAN, optomisers, as per section 2.1, and other traits like depth, learning rate etc. - Genes/traits from better performing probes should be given a higher "survival" probability

This should comprise a two part genome of sub-experts and general traits.



## General Section 2 rules

All datasets and metrics should be identical to those used in section 1.3. This is so that the utility of each section can be fairly benchmarked against the basic KAN architecture.