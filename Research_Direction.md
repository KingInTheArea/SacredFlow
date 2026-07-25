

### ***What the research community has done***

* The perspective problem in crowd counting is well-studied, but the dominant academic approach is fundamentally different from yours. Most existing methods use multi-column networks with different receptive field sizes — large, medium, and small convolution kernels in separate columns — to handle people at three distances: near, middle, and far, then merge the resulting density maps. More recent work like PANet proposes dynamic receptive fields that adapt based on density distributions — in far-end regions where density variation is larger, smaller overlapping receptive fields are used, while near-end regions use larger receptive fields for more context. arxivTheCVF
* Context-Aware Crowd Counting (CVPR 2019) explicitly argues that standard convolutions implicitly use the same receptive field everywhere, which is wrong for perspective-distorted scenes where scale varies continuously across the image — and that prior patch-based or multi-filter approaches indiscriminately fuse information at all scales without accounting for this. arxiv

### ***Where my approach is different***

All of the above are pure density estimation methods — they count people but don't track identities. Your system's key distinction is the hybrid detection + density split: you use YOLO11x + DeepSORT in the near field specifically to get identity-persistent tracking (not just a count), which enables the dwell-time/darshan-time measurement. No pure density estimation method can give you that.

The closest academic parallel to your spatial-split idea is work that proposes adaptive multi-scale far-and-near distance networks that explicitly distinguish between near and far features, allocating different receptive fields based on distance from the camera. But this is still a single unified model, not a deliberate pipeline switch between detection and density estimation. MDPI



 "***Prior work handles perspective by adapting a single model's receptive fields. Our approach makes a harder architectural choice — switching between fundamentally different model types (detection vs. density estimation) at a spatial boundary — which sacrifices elegance for a practical gain: identity tracking in the near field, which no density-only method provides. The trade-off is a hard-coded split ratio rather than a learned, continuous perspective correction***."


###### **What MCNN actually does**

MCNN (Multi-Column CNN, Zhang et al. 2016) uses three parallel columns with large, medium, and small convolution kernels, processes the entire image through all three columns simultaneously, then merges the density maps. Crucially — it applies all three columns to every pixel. There's no spatial boundary learned; instead, each column implicitly specializes across the whole image through training. The "split" is in kernel size/receptive field, not in spatial location.

Why it's not quite a learned split boundary

A true learned spatial boundary would be something like a network that decides per-image or per-region which model to route a spatial patch to, based on estimated depth/density. MCNN doesn't do that — it fuses everything globally, which is actually why later work like PANet criticized it: the receptive fields are fixed for all images regardless of actual perspective geometry.

What actually is closest to a learned spatial boundary

Two better answers for that specific question:



Switching CNN (Sam et al., 2017) — explicitly trains a classifier to select, per image patch, which density regressor to apply. That's genuinely the closest thing to a learned routing/switching mechanism across spatial regions.

Perspective-Guided Convolution (PGCNet, Yan et al., ICCV 2019) — uses an explicit perspective map to spatially modulate convolution behavior, which is essentially a continuous learned version of your hard-coded split ratio.



How to use this in an interview

A sharper answer to "how would you make the split boundary learned?" would be:



"Something like Switching CNN's patch-level classifier, but instead of selecting a density regressor, selecting between a detection-tracking branch and a density estimation branch based on estimated local depth or density. PGCNet shows you can encode perspective geometry explicitly into the network — combining that with a detection/density router would be the natural extension."



That answer shows you know MCNN, know why it's not quite the right reference, and have thought one step beyond the obvious. That's exactly the kind of nuance that lands well in a research interview.





### Switch-CNN (CVPR 2017) — Sam, Surya, Babu

##### The core idea

Patches from a grid within a crowd scene are relayed to independent CNN regressors based on crowd count prediction quality of the CNN established during training. The independent CNN regressors are designed to have different receptive fields, and a switch classifier is trained to relay each crowd scene patch to the most appropriate regressor. GitHub

**In plain terms**
the image is divided into a grid of patches. A separate "switch" classifier network looks at each patch and decides which density regressor (small, medium, or large receptive field) is best suited for that patch's density level — then routes it there. The final density map is assembled from whichever regressor handled each patch.

###### Why this matters relative to MCNN

MCNN applies all three columns to the whole image simultaneously and merges everything at the end — it has no spatial routing intelligence. Switch-CNN adds an explicit decision layer that says "this patch is densely packed, route it to the small-receptive-field regressor; this patch is sparse, route it to the large one." It switches between networks automatically to find the best one for each crowd density, improving accuracy across all situations. GitHub

###### The limitation relevant to your work

The switch classifier routes between density regressors only — all branches are still pure density estimation, producing no identity information. Also, the switching is patch-level and learned on crowd density labels, not on geometric perspective maps. It doesn't explicitly know where in the image near vs. far field is — it infers it implicitly from density statistics.

Relation to SacredFlow

Your SPLIT\_RATIO = 0.4 is a hand-coded, geometry-aware version of what Switch-CNN's classifier does implicitly. You make a harder architectural choice: route to a fundamentally different model type (detector vs. density estimator) rather than between density regressors of different scales. The upside is identity tracking. The downside is your boundary is fixed by a human rather than learned from data.



### PGCNet (ICCV 2019) — Yan, Yuan, Zuo et al.

#### The core idea

PGCNet utilizes perspective information to guide the spatially variant smoothing of feature maps before feeding them to the successive convolutions. An effective perspective estimation branch is also introduced to PGCNet, which can be trained in either supervised setting or weakly-supervised setting when the branch has been pre-trained. GitHub

Rather than splitting the image into zones or columns, PGCNet keeps a single-column architecture but modulates how convolution behaves at each spatial location based on an estimated perspective map. The perspective map tells the network: "at this pixel location, people should appear this large" — and the convolution's smoothing kernel is adjusted accordingly, continuously across the image.

##### Why it directly addresses your limitation

Most state-of-the-arts adopt multi-scale or multi-column architectures to address the perspective issue, but they generally fail in modeling continuous scale variations since only discrete representative scales are considered. This is exactly the criticism of your hard-coded SPLIT\_RATIO — it creates a discrete boundary (40%/60%) rather than a continuous, geometry-aware transition. PGCNet solves this by making the perspective correction spatially continuous and learned. arxiv

The weakly-supervised angle

A notable practical feature: the perspective estimation branch can be trained in either supervised setting or weakly-supervised setting when the branch has been pre-trained — meaning you don't necessarily need explicit perspective map annotations (which are expensive to collect) to use it. This is relevant for your setting since you have no ground-truth labels for your temple footage. GitHub

Relation to SacredFlow

PGCNet is the most direct academic answer to "how would you make the split boundary learned?" — it shows you can encode perspective geometry into the network continuously, without a hard spatial cut. The natural extension of SacredFlow toward PGCNet's approach would be: instead of a fixed horizontal split, train a lightweight perspective estimation branch on your temple footage (possibly weakly supervised), then use its output to decide per-region which pipeline to route to.



The one-paragraph interview summary if you need to connect all three papers to your own work:



"MCNN showed you could handle perspective by using multiple receptive field sizes in parallel columns — but it applies all columns to every pixel without any spatial routing intelligence. Switch-CNN improved on this by adding a learned patch-level classifier that routes each image region to the best-suited density regressor. Our approach takes a similar routing intuition but makes a harder architectural choice — we route spatially between a detection-tracking pipeline and a density estimation pipeline, rather than between density regressors, which is what enables identity tracking. The limitation is our boundary is hand-coded at 40%, not learned. PGCNet is the most principled resolution to that — it encodes perspective geometry directly into the convolution operation, enabling continuous, learned, spatially-varying scale correction without any hard boundary at all. Extending SacredFlow toward PGCNet's perspective estimation branch would be the natural next step."

