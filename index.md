# Optical flow with OpenCV and CUDA

<p align="center">
  <img src="opticalFlowChapter.jpg" width="400">
</p>

Processing a static image of a scene has application in different fields of technology and a concrete example has been provided at [Face recognition using OpenCV and CUDA](https://vitalitylearning2021.github.io/faceRecognitionOpenCVCUDA/) with face recognition.  
In the case when a sequence of consecutive images of the same scene are available, acquired by a single or many fixed or moving cameras, it is possible to extract information on the motion of the objects of interest. Indeed, the information contained in the motion covers a role of primary importance in many computer vision problems like the tracking
of moving targets, the temporal prediction of the arrangement of the surrounding environment, the recovery of the three-dimensional velocity field of image points or the segmentation of the image in parts corresponding to moving targets. Moreover, motion information is an indispensable prerequisite for tasks finalized to the movement of
autonomous robots and to their interaction with the environment.  
In this project, we will deal with *optical flow* techniques, namely, with schemes estimating a velocity field from the only intensity of image sequences. However, since the images acquired by a camera convey two-dimensional information, then, by optical flow, it is not possible to reconstruct the full three-dimensional velocity field, but we have to settle for its projection, precisely, on a two-dimensional space. Nonetheless, even a two-dimensional projection of the three-dimensional velocity field brings valuable information in many applications. For the sake of simplicity, we will limit ourselves to the case of a single, still camera.  
More in detail, we will consider four robust, reliable and fast approaches for the accurate estimation of both dense and sparse optical flow: the Farnebäck method, the Brox *et al.* scheme, the Total-Variation-L1 (TV-L1) regularization and the Lucas-Kanade approach. While, in [Face recognition using OpenCV and CUDA](https://vitalitylearning2021.github.io/faceRecognitionOpenCVCUDA/), we used CUDA-accelerated OpenCV routines to set up face recognition along with the help of cuSOLVER, in
this project we will discover how the mentioned techniques are actually already implemented in OpenCV. Nevertheless, although off-the-shelf, the rationale of the dealt with algorithms will be recalled since a conscious use of what is inside any library is always much better than an usage of the same routines as black-boxes. A conscious use of the
libraries indeed enables a consistent definition of the input parameters and a knowledge of the algorithm limitations.  
In the present project, we will deal with:

  - Dense and sparse optical flow;
  - Farnebäck optical flow;
  - Brox *et al.* optical flow;
  - Total-Variation-L1 regularized optical flow;
  - Lucas-Kanade optical flow;
  - Color coding and streamlines drawing for optical flow representation;
  - Optical flow with CUDA-accelerated OpenCV.

## Getting started

In the present project, the key elements of optical flow techniques needed for a conscious use of CUDA-accelerated OpenCV routines will be given. No particular prerequisite is necessary.  
In the next section, we introduce the concepts of optical flow and motion fields.  
The optical flow concept will be qualitatively illustrated starting from its introduction in the field of experimental psychology. Later on, the motion field concept will be quantitatively described as the projection of the three-dimensional velocity field in the image plane. Finally, the relation between optical flow and motion fields will be discussed by highlighting how quantitatively optical flow is an approximation of motion fields

## Optical flow and motion fields

In the present section, we will introduce the concept of optical flow starting from its very origins in the field of applied psychology. The motivation for its introduction provides a valuable example to understand the usefulness of optical flow. Later on, we will more formally describe the concept of motion fields and highlight how quantitatively optical flow is able to provide, in general, only an approximation thereof. We will also point out how, in some remarkable cases, optical flow can lead to misleading conclusions about motion fields.

### The concept of optical flow

The concept of optical flow draws its origin in the field of experimental psychology, that is, in a framework significantly different from that of the technical-engineering applications we are referring to. It was a result of the research of J.J. Gibson which was appointed, during World War II, the task of understanding how pilots fly according to what they see. While studying the perception of moving objects, he introduced the concept of *optical flow* as :

> Optical flow is the distribution of apparent velocities of movement of brightness patterns in an image. Optical flow can arise from the relative motion of objects and the viewer.

To illustrate the concept of optical flow, it is convenient to exploit the same, although dated, three images used by Gibson.  
Figure [1](#flyby) below illustrates the optical flow associated to a flight over a landing field. The arrows describe the motion of the scene points as they appear to the pilot according to the definition above. In particular, the arrows represent the motion direction, through their pointing directions, and the speed, through their lengths, of some scene points.

<p align="center">
  <img src="flyby.png" width="400" id="flyby">
  <br>
     <em>Figure 1. Flyby optical flow image.</em>
</p>

Figure [1](#flyby) actually illustrates a flyby: the aircraft is passing over the airfield, but it is not landing. We deduce this from the fact that all the arrows point from a position on the horizon and not from the landing area.  
Next figure [2](#landing) displays a landing.

<p align="center">
  <img src="landing.png" width="400" id="landing">
  <br>
     <em>Figure 2. Landing optical flow image.</em>
</p>

Now, the arrows point from a position inside the landing area. When a pilot is landing, the point of the landing area towards which he is moving appears standing still in his field of view. Opposite to that, the other points of the surrounding visual environment move away from such a point.  
Finally, figure [3](#flybyLeft) below represents a flyby to the left.

<p align="center">
  <img src="flybyLeft.png" width="400" id="flybyLeft">
  <br>
     <em>Figure 3. Flyby to the left.</em>
</p>

It reproduces a scenario in which the pilot is looking to the side and, in particular, over the right shoulder as the plane flies to the left.  
In all the three cases, the vector fields express the optical flow. Notice that the shown images do not mean to faithfully reproduce the *motion field*, i.e., the projections of the velocity vectors of the three-dimensional space into the image plane (a two-dimensional space). Nevertheless, they provide the impression of how movement is occurring.  
According to Gibson, an approximation of the motion field provides the pilot of non-ambiguous information concerning his speed, altitude and direction, namely, a sufficient information to determine the movement of the observer with respect to the surrounding environment.  
In next subsection, we turn to discussing two-dimensional motion fields.

### Two-dimensional motion fields

The two-dimensional motion field problem is illustrated in figure [4](#apertureProblem):

<p align="center">
  <img src="apertureProblem.png" width="400" id="apertureProblem">
  <br>
     <em>Figure 4. Illustrating the two-dimensional motion fields.</em>
</p>

The above figure illustrates a point <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}=(X, Y, Z)"> that moves over a certain trajectory <img src="https://render.githubusercontent.com/render/math?math=T"> with speed <img src="https://render.githubusercontent.com/render/math?math=\mathbf{V}=(V_x,V_y,V_z)">. The motion of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}"> is observed by a point-like camera located at <img src="https://render.githubusercontent.com/render/math?math=O">, forming an image of the scene over the plane <img src="https://render.githubusercontent.com/render/math?math=S">. Point <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}"> must be thought of as one of the many points forming an object moving behind <img src="https://render.githubusercontent.com/render/math?math=S"> and of which we isolate <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}"> to illustrate the concept of two-dimensional motion fields. For the sake of simplicity, the camera is further assumed to be still, although, in principle, it could be also
dealt with as a moving camera. The plane <img src="https://render.githubusercontent.com/render/math?math=S"> represents the focal plane of the camera and, again for simplicity and without losses of generality, it is assumed to coincide with the plane <img src="https://render.githubusercontent.com/render/math?math=Z=1">. The point <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}"> appears to the camera as being “projected” over <img src="https://render.githubusercontent.com/render/math?math=S"> at a point <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}=(x,y,1)">, where

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\left\{
                \begin{array}{ll}
                  x=\frac{X}{Z}\\
                  y=\frac{Y}{Z}
                \end{array}
              \right.." id="xxx"> [1]
</p>

Point <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}"> will move, due to the motion of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}">, with a certain speed <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}=(v_x,v_y,0)">. Of course, the <img src="https://render.githubusercontent.com/render/math?math=z">-component of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> vanishes since
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}"> is constrained to move on <img src="https://render.githubusercontent.com/render/math?math=Z=1">.  
As for <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}">, there will be many other points on the <img src="https://render.githubusercontent.com/render/math?math=Z=1"> plane that will be the images of bodies moving behind <img src="https://render.githubusercontent.com/render/math?math=S"> and having a certain velocity <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}">. Of course, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> will be different from point to point. The collection of all point velocities <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> is a *two-dimensional motion field*.  
It should be noticed that all the up to now considered quantities <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{p}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{V}"> and
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> are time-dependent.  
To determine the relation between <img src="https://render.githubusercontent.com/render/math?math=\mathbf{V}"> and
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}">, let us first observe that

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}=Z\mathbf{p}." id="relationPp"> [2]
</p>

On taking the time derivative of [\[2\]](#relationPp), we have

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\mathbf{V}=V_z\mathbf{p} %2B Z\mathbf{v}." id="relationVv"> [3]
</p>

For discussion purposes, eq. [\[3\]](#relationVv) should be regarded as an equation in which <img src="https://render.githubusercontent.com/render/math?math=\mathbf{V}"> represents the unknown. In other words, we want to determine the velocity field of the points moving behind <img src="https://render.githubusercontent.com/render/math?math=S">. The known quantities in eq. [\[3\]](#relationVv) are available at the right-hand side, since we assume to have <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> available in some way. Unfortunately, eq. [\[3\]](#relationVv) has only two useful components since the projection of both sides on <img src="https://render.githubusercontent.com/render/math?math=z"> is the identity <img src="https://render.githubusercontent.com/render/math?math=V_z=V_z">. As a consequence, eq. [\[3\]](#relationVv) tells us that, even following an estimate of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}">, we could not be able to estimate the velocities of the points <img src="https://render.githubusercontent.com/render/math?math=\mathbf{P}">. Therefore, we can conclude that, following the observation of a moving object from a still camera, it is not possible to extract the motion of the scene points. Our only hope is to estimate the velocity field <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> to have a rough, although incomplete, idea of the motion of the points behind <img src="https://render.githubusercontent.com/render/math?math=S">.  
Reinforcing the issue that, from two-dimensional motion field, the actual motion of bodies behind <img src="https://render.githubusercontent.com/render/math?math=S"> cannot be deduced, it should be also remarked that an approximation to <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> is typically provided by what we have defined as optical flow. In next subsection, we underline how the very estimate of motion fields from optical flows can lead to erroneous conclusions in a certain number of canonical cases.

### The slippery relation between optical flows and motion fields

From what illustrated in figures [1](#flyby), [2](#landing) and [3](#flybyLeft), the optical flow can be quantitatively defined as the motion of the brightness pattern of an image. Unfortunately, following such a definition, optical flows do not necessarily coincide with motion fields. In this subsection, we present three classical examples to
convince ourselves of this fact:

1.  the case of a sphere rotating about itself with fixed illumination;
2.  the case of a still sphere illuminated by a moving source;
3.  the Barber’s shop pole.

These three cases should be used as disclaimers on the possibility of consistently using optical flows to estimate motion fields. However, our underlining assumption is that, in many cases and apart from the degenerate ones we will shortly present, we reasonably expect for the sequel that the optical field is close to the motion field.  
Let us now show the first degenerate case.

#### A sphere rotating about itself with fixed illumination

The following figure shows a sphere illuminated by a static source.

<p align="center">
  <img src="rotatingSphere.png" width="400" id="rotatingSphere">
  <br>
     <em>Figure 5. Sphere rotating about itself with fixed illumination.</em>
</p>

The source in figure [5](#rotatingSphere) generates a certain light intensity distribution over the sphere. If the sphere rotates about itself, thanks to its symmetry, such shading pattern will not change. As long as the sphere rotates about itself, then, the brightness pattern of its image remains unchanged. Since the optical flow evaluates, as
mentioned, the motion of the brightness pattern, then each point will appear still. The optical flow would lead then to the wrong conclusion that the sphere is motionless.  
It is now the turn of the second degenerate case.

#### A still sphere illuminated by a moving source

Figure [6](#rotatingSource) below shows a sphere this time still, but with a moving source.

<p align="center">
  <img src="rotatingSource.png" width="400" id="rotatingSource">
  <br>
     <em>Figure 6. Still sphere illuminated by a moving source.</em>
</p>

As long as the source moves, the lighting source changes so that the shading pattern changes and, accordingly, in this case, the optical flow is not zero when the motion field is vanishing. Accordingly, one would erroneously conclude that the sphere is moving while it is not.  
Finally, let us present the last degenerate case.

#### The Barber’s shop pole

The rotating Barber’s pole is the last example in which we show how the optical flow can be a totally erroneous approximation of motion fields. The following figure illustrates the rotating Barber’s shop pole problem.

<p align="center">
  <img src="barbersPole.png" width="400" id="barbersPole">
  <br>
     <em>Figure 7. Rotating Barber’s pole.</em>
</p>

The left side of figure [7](#barbersPole) illustrates the pole: while rotating, the pole’s strips move right. The central side of figure [7](#barbersPole) shows the actual strips motion: the arrows point to the right. However, as it can be understood from the right side of figure [7](#barbersPole), the strips appear moving upwards from one
frame to the next, so that the optical flow represents a totally erroneous estimate of motion field again.  
  
In the just shown examples, we have assumed to be able to compute the optical flow. Even by assuming a full knowledge of the optical flow, we have seen how erroneous conclusions can be reached.  
In the next section, we aim at pointing out the difficulties in the computation of the optical flow.

## The optical flow constraint equation and the aperture problem

The time has become for beginning the discussion on the computation of the optical flow by introducing the *optical flow constraint equation*. We will underline, by illustrating the *aperture problem*, how, strictly speaking, estimating both the components of the velocities of the points in an image is not possible and the consequences of this.  
Let us start with the introduction of the optical flow constraint equation. To this end, let us regard a sequence of images changing in time as they were a movie, by watching at the points altogether. In other words, we adopt what in fluid mechanics is called a *Lagrangian approach*. In this case, the image intensity <img src="https://render.githubusercontent.com/render/math?math=I"> depends on both the above introduced space coordinates <img src="https://render.githubusercontent.com/render/math?math=(x,y)"> and on time <img src="https://render.githubusercontent.com/render/math?math=t">. In other words, <img src="https://render.githubusercontent.com/render/math?math=I=I(x,y,t)">.  
Let us now change point of view. Let us consider a specific point and follow it through the scene, by getting unconcerned about all the other points. In other words, let us follow the motion <img src="https://render.githubusercontent.com/render/math?math=(x(t),y(t)"> by adopting what, in fluid mechanics, is called *Eulerian approach*. In
this case, we regard the intensity as <img src="https://render.githubusercontent.com/render/math?math=I=I(x(t),y(t),t)=I(t)">. Saying it differently, we regard the intensity as a function of time since we are following a point moving along the <img src="https://render.githubusercontent.com/render/math?math=(x(t),y(t))"> trajectory. Now, if the time increases from <img src="https://render.githubusercontent.com/render/math?math=t"> to <img src="https://render.githubusercontent.com/render/math?math=t %2B dt">, the point that we are following moves from <img src="https://render.githubusercontent.com/render/math?math=(x(t),y(t))"> to <img src="https://render.githubusercontent.com/render/math?math=(x(t) %2B v_x dt,y(t) %2B v_y dt)"> due to its velocity. In principle, also the intensity of the followed point might change from <img src="https://render.githubusercontent.com/render/math?math=I(t)"> to <img src="https://render.githubusercontent.com/render/math?math=I(t) %2B \frac{dI}{dt}(t)dt">. If we
assume that the intensity of the followed point does not change, namely, we adopt the *constant brightness assumption*, then we have

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt}=0." id="constantBrightness"> [4]
</p>

On using the chain rule, equation [\[4\]](#constantBrightness) implies that

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt}=\frac{\partial I}{\partial x}v_x %2B \frac{\partial I}{\partial y}v_y %2B \frac{\partial I}{\partial t}=0." id="constantBrightnessExpanded"> [5]
</p>

Equation [\[5\]](#constantBrightnessExpanded) expresses the so called *optical flow constraint* equation. In such an equation, the partial temporal derivative of <img src="https://render.githubusercontent.com/render/math?math=I"> must be considered as a datum since we know the temporal evolution of the images. Analogously,
the partial derivatives of <img src="https://render.githubusercontent.com/render/math?math=I"> with respect to <img src="https://render.githubusercontent.com/render/math?math=x"> and <img src="https://render.githubusercontent.com/render/math?math=y"> must be assumed to be known since we know the individual images. Opposite to that, the components along <img src="https://render.githubusercontent.com/render/math?math=x"> and <img src="https://render.githubusercontent.com/render/math?math=y"> of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> must be considered to be unknown. The question thus arises on whether the optical flow can be estimated under the constant brightness assumption.  
To investigate on whether this is possible, we illustrate the so called *aperture problem* and note that equation [\[5\]](#constantBrightnessExpanded) is a single equation in two unknowns, so that it is under-determined. If we rewrite equation [\[5\]](#constantBrightnessExpanded) as

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\nabla}\cdot \mathbf{v} %2B \frac{\partial I}{\partial t}=0," id="normalFlow"> [6]
</p>

then we deduce that it is possible to recover only the component of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> along the direction of the gradient <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\nabla}I"> which, as mentioned, is known. In other words, from equation [\[6\]](#normalFlow), we can only determine the so called *normal flow*, i.e., the flow along the direction of the image gradient, but we cannot determine the flow along the tangent direction of the iso-intensity contour, i.e., along the direction perpendicular to the image gradient. This expresses the so called *aperture problem*.  
The aperture problem is illustrated in figures [8](#actualMotion) and [9](#perceivedMotion). In particular, in figure [8](#actualMotion) below, a circular aperture and two blue lines moving in a certain direction are illustrated.

<p align="center">
  <img src="actualMotion.png" width="400" id="actualMotion">
  <br>
     <em>Figure 8. Aperture problem: actual motion of two
lines.</em>
</p>

The circular aperture represents all that is visible in the scene. It should be noticed that, in figure [8](#actualMotion), the blue lines are depicted also outside the circle, but it should be understood that the only observable thing is the portion of the lines within the circle.  
Figure [8](#actualMotion) illustrates the actual motion. Actually, due to the aperture problem, the perceived motion is different and illustrated in figure [9](#perceivedMotion) below.

<p align="center">
  <img src="perceivedMotion.png" width="400" id="perceivedMotion">
  <br>
     <em>Figure 9. Aperture problem: perceived motion of two
lines.</em>
</p>

As it can be seen from the above figure and also from figure [8](#actualMotion), the blue lines are the iso-intensity lines and the perceived motion is orthogonal to such lines as predicted by equation [\[6\]](#normalFlow).  
Up to now, we have seen how the motion of the points behind the image plane of a camera can be represented by the motion fields, within certain limits. Moreover, we have underlined how the motion field can be approximated by the optical flow, again within certain limits. Finally, we have pointed out what are the difficulties in the calculation of the optical flow.  
The under-determinacy of the optical flow constraint equation has pushed the literature towards the development of several methods to solve this problem. Some of them will be dealt with in this chapter and will be subject of implementation in OpenCV accelerated with CUDA.  
Before proceeding further with the description of the implemented methods for the optical flow estimate, let us organize the optical flow estimation methods into *dense* and *sparse* methods.

## Dense and sparse optical flow methods

Dense methods process all the pixels of an image, while, in sparse methods, only some pixels are subject of processing. In this case, we only consider the motion of image features like corners, ridges, edges and blobs. As we will see in the following, optical flow methods exploit local pixel information, so that many of them can be used both for dense and sparse optical flow.  
Below, as dense methods, we will consider the Farneb<span>ä</span>ck’s approach, the Brox *et al.* method, the dense iterative Lucas-Kanade method with pyramids and the Zach, Pock and Bischof Dual TV-L1 Optical Flow method . Furthermore, as sparse method, we will consider the implementation for a sparse feature set of the iterative Lucas-Kanade
scheme with pyramids.  
In next section, we start recalling dense optical using Farneb<span>ä</span>ck’s approach.

<p align="center" id="functionalFarneback" >
</p>
## Theory: Dense optical flow using Farneb<span>ä</span>ck’s approach

The idea behind Farneb<span>ä</span>ck’s approach  is to approximate an image in a neighborhood of each pixel having indices <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> with a quadratic polynomial. In other words,

<p align="center">
  <img src="equation_7.png" width="300" id="farnebackApproximation">     [7]
</p>
    
where <img src="https://render.githubusercontent.com/render/math?math=G"> is the image, <img src="https://render.githubusercontent.com/render/math?math=(\Delta m,\Delta n)"> represents the offset with respect to the <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> reference pixel, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{I}^t=[\Delta m, \Delta n]">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_1"> is a <img src="https://render.githubusercontent.com/render/math?math=2\times 2"> symmetric matrix, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_1"> is a vector and <img src="https://render.githubusercontent.com/render/math?math=c_1"> is a scalar. We stress that <img src="https://render.githubusercontent.com/render/math?math=(\Delta m,\Delta n)"> does not represent a movement, but only serves to describe a neighborhood of the <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> pixel in the *same* image <img src="https://render.githubusercontent.com/render/math?math=G">.  
In eq. [\[7\]](#farnebackApproximation), <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_1">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_1"> and <img src="https://render.githubusercontent.com/render/math?math=c_1"> can be estimated using a weighted least squares approach. Obviously, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_1">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_1"> and <img src="https://render.githubusercontent.com/render/math?math=c_1"> are functions of <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> since the image could show a different expansion around different pixels. In order to keep notation simple, in the first part of this section, we dismiss the explicit dependence of such quantities on <img src="https://render.githubusercontent.com/render/math?math=(m,n)">.  
On denoting by <img src="https://render.githubusercontent.com/render/math?math=H"> the image acquired immediately after <img src="https://render.githubusercontent.com/render/math?math=G"> and by following the same approach of the quadratic expansion around <img src="https://render.githubusercontent.com/render/math?math=(m,n)">, we have

<p align="center">
  <img src="equation_8.png" width="300" id="farnebackApproximationH">     [8]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_2">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_2"> and <img src="https://render.githubusercontent.com/render/math?math=c_2"> have the same meaning of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_1">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_1"> and <img src="https://render.githubusercontent.com/render/math?math=c_1"> in equation [\[7\]](#farnebackApproximation).  
In the case when the neighborhood of the pixel <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> of <img src="https://render.githubusercontent.com/render/math?math=G"> moves rigidly of a quantity <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}=(d_m,d_n)"> to image <img src="https://render.githubusercontent.com/render/math?math=H">, then

<p align="center">
  <img src="equation_9.png" width="400" id="relationHG">     [9]
</p>

By taking into account relation [\[9\]](#relationHG) as well as expansions [\[7\]](#farnebackApproximation) and [\[8\]](#farnebackApproximationH), then we have the following links

<p align="center">
  <img src="equation_10.png" width="200" id="linksHG">     [10]
</p>

We skip the details for obtaining [\[10\]](#linksHG). It should be anyway noted that the second equation of [\[10\]](#linksHG) provides the following relation

<p align="center">
  <img src="equation_11.png" width="100" id="equationInD">     [11]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\Delta \mathbf{b}=-0.5(\mathbf{b}_2-\mathbf{b}_1)">. Solving such an equation will provide the translation <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}"> but is possible provided that <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_1"> is invertible.  
However, it should be taken into account that the translation of the neighborhood of the pixel <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> from <img src="https://render.githubusercontent.com/render/math?math=G"> to <img src="https://render.githubusercontent.com/render/math?math=H"> is not necessarily ideal and that, accordingly, the links [\[10\]](#linksHG) are not necessarily met. In this case, we can only say, for example, that <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_2\simeq \mathbf{A}_1">. For this reason, Farneb<span>ä</span>ck rewrites equation [\[11\]](#equationInD) by the average of the two matrices, namely as

<p align="center">
  <img src="equation_12.png" width="100" id="equationInDRewritten">     [12]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\Delta \mathbf{A}=0.5(\mathbf{A}_1 %2B \mathbf{A}_2)">. As above underlined, matrices <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{A}_2">
as well as the vectors <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}_2"> depend on the point <img src="https://render.githubusercontent.com/render/math?math=(m,n)">. For this reason, equation [\[12\]](#equationInDRewritten) should be solved pointwise thus obtaining <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}(m,n)">.  
In order to prevent noisy solutions and to provide a smoothing effect, Farneb<span>ä</span>ck reformulates the search for the solution in <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}"> of equation [\[12\]](#equationInDRewritten) as a least-square problem by integrating it over a neighborhood of each pixel. Accordingly, the displacement <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}"> is searched for as the minimum of the functional

<p align="center">
  <img src="equation_13.png" width="400" id="functionalFarneback">     [13]
</p>

where <img src="https://render.githubusercontent.com/render/math?math={\mathcal I}(m,n)"> defines a neighborhood of the pixel <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> and <img src="https://render.githubusercontent.com/render/math?math=w"> is a properly chosen window function. In [\[13\]](#functionalFarneback), the dependence of <img src="https://render.githubusercontent.com/render/math?math=\Delta \mathbf{A}"> and <img src="https://render.githubusercontent.com/render/math?math=\Delta \mathbf{b}"> from <img src="https://render.githubusercontent.com/render/math?math=(m^\prime,n^\prime)">, namely, the pixels of the neighborhood of <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> has been explicitly indicated. For what already said, the
minimization of [\[13\]](#functionalFarneback) must be repeated pixel-by-pixel.  
In next section, we lead the discussion of the Brox *et al.*’s approach.

## Theory: Dense optical flow using Brox *et al.*’s approach

Brox *et al.*’s approach  originates from the observation that the grey value consistency assumption behind equation [\[6\]](#normalFlow) is not always met and that, as long as
the points move, their respective brightness is susceptible, in natural scenes, to slight changes. Accordingly, the invariance understood in equation [\[4\]](#constantBrightness) is placed side by side to a further invariance criterion. In particular, the gradient of the image grey value is assumed not to vary along the displacement. In other words,

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\nabla}I(x,y,t)=\mathbf{\nabla}I(x %2B d_x,y %2B d_y,t %2B 1)," id="xxx">       [14]
</p>

where we have assumed that time increases of <img src="https://render.githubusercontent.com/render/math?math=1"> unit from frame to frame and that <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}(x,y)=(d_x(x,y),d_y(x,y))"> is the displacement of point <img src="https://render.githubusercontent.com/render/math?math=(x,y)">.  
The idea of Brox *et al.* consists of searching for <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}(x,y)"> by a least-square minimization. One possibility would be to minimize the functional

<p align="center">
  <img src="equation_15.png" width="700" id="broxFunctional0">     [15]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\Omega(x,y)"> represents a neighborhood of point <img src="https://render.githubusercontent.com/render/math?math=(x,y)"> and <img src="https://render.githubusercontent.com/render/math?math=\gamma"> relatively weights both the assumptions and establishes the trade off between the two enforced invariance conditions. As for the Farneb<span>ä</span>ck’s approach, the minimization of functional
[\[15\]](#broxFunctional0) must be performed pointwise.  
In order to reduce the influence of outliers pixels, an increasing concave function <img src="https://render.githubusercontent.com/render/math?math=\Psi(s)"> is applied leading to

<p align="center">
  <img src="equation_16.png" width="700" id="broxFunctional1">     [16]
</p>

A possible choice for <img src="https://render.githubusercontent.com/render/math?math=\Psi(s)"> is <img src="https://render.githubusercontent.com/render/math?math=\Psi(s)=\sqrt{s^2 %2B \epsilon^2}">, where <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is a small constant, for example, <img src="https://render.githubusercontent.com/render/math?math=\epsilon=0.01">.  
A smoothness condition to account for a piecewise smooth flow field is also enforced by adding the following smoothing penalizer

<p align="center">
  <img src="equation_17.png" width="300" id="xxx">     [17]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\nabla}_3=(\partial/\partial x, \partial/\partial y, \partial/\partial t)^t">
is a spatio-temporal gradient.  
Overall, the following functional is minimized:

<p align="center">
  <img src="equation_18.png" width="300" id="broxTwoTerms">     [18]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is a *regularization parameter*. Larger values of <img src="https://render.githubusercontent.com/render/math?math=\alpha"> will lead to smoother solutions.  
The minimizer of [\[18\]](#L1Functional3) must satisfy the following Euler-Lagrange equations 

<p align="center">
  <img src="equation_19.png" width="700" id="eulerLagrange">     [19]
</p>

with reflecting boundary conditions, where

<p align="center">
  <img src="equation_20.png" width="700" id="xxx">     [20]
</p>

Equations [\[19\]](#eulerLagrange) are highly non-linear and can be solved by fixed point iterations. In other words, letting <img src="https://render.githubusercontent.com/render/math?math=(d_x^k,d_y^k)"> the unknown value at the <img src="https://render.githubusercontent.com/render/math?math=k">-th iterations, the first equation of [\[19\]](#eulerLagrange) is rewritten as

<p align="center">
  <img src="equation_21.png" width="900" id="fixedPoint">     [21]
</p>

where

<p align="center">
  <img src="equation_22.png" width="500" id="xxx">     [22]
</p>

and <img src="https://render.githubusercontent.com/render/math?math=du^k"> and <img src="https://render.githubusercontent.com/render/math?math=dv^k"> are the updates for <img src="https://render.githubusercontent.com/render/math?math=d_x^k"> and <img src="https://render.githubusercontent.com/render/math?math=d_y^k", respectively. The second equation of [\[19\]](#eulerLagrange) can be written similarly.  
Unfortunately, equation [\[21\]](#fixedPoint) is still non-linear in <img src="https://render.githubusercontent.com/render/math?math=du^k"> and <img src="https://render.githubusercontent.com/render/math?math=dv^k"> since such variables are contained also in <img src="https://render.githubusercontent.com/render/math?math=A^k"> and <img src="https://render.githubusercontent.com/render/math?math=B^k">. In order to remove the remaining non-linearity, a second, inner, fixed point iteration is operated. In particular, on denoting by <img src="https://render.githubusercontent.com/render/math?math=A^{k,l}">, <img src="https://render.githubusercontent.com/render/math?math=B^{k,l}">, <img src="https://render.githubusercontent.com/render/math?math=du^{k,l}"> and <img src="https://render.githubusercontent.com/render/math?math=dv^{k,l}"> the values of the corresponding variables at the <img src="https://render.githubusercontent.com/render/math?math=l">-th inner iteration, then equation [\[21\]](#fixedPoint) is
rewritten as:

<p align="center">
  <img src="equation_24.png" width="800" id="fixedPoint2">     [24]
</p>

so that, keeping <img src="https://render.githubusercontent.com/render/math?math=A^{k,l}"> and <img src="https://render.githubusercontent.com/render/math?math=B^{k,l}"> fixed at the <img src="https://render.githubusercontent.com/render/math?math=l">-th inner iteration, the equation is now linear in <img src="https://render.githubusercontent.com/render/math?math=du^{k,l %2B 1}"> and <img src="https://render.githubusercontent.com/render/math?math=dv^{k,l+1}">. Equation [\[24\]](#fixedPoint2), along with its non-written companion, form a linear system of equations in <img src="https://render.githubusercontent.com/render/math?math=du^{k,l+1}"> and <img src="https://render.githubusercontent.com/render/math?math=dv^{k,l %2B 1}"> which can be solved by a standard iterative method.

## Theory: Dense optical flow using TV-L1 regularization

We have already underlined that equation [\[6\]](#normalFlow) is under-determined, being a single equation in two unknowns. In order to solve it, the approach by Horn and Schunck , which has been much popular for a while, consists of placing a smoothness condition enforcing the displacement <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}(x,y)"> to be regular side by side to equation [\[6\]](#normalFlow). In particular, the
Horn and Schunck’s approach determined <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}(x,y)"> by the minimization of the following functional:

<p align="center">
  <img src="equation_25.png" width="500" id="L1Functional1">     [25]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\Omega(x,y)"> represents again a neighborhood of <img src="https://render.githubusercontent.com/render/math?math=(x,y)"> and <img src="https://render.githubusercontent.com/render/math?math=\alpha">, like <img src="https://render.githubusercontent.com/render/math?math=\gamma"> for the Brox *et al.*’s approach, weights both the assumptions.  
If we replace the first term of [\[25\]](#L1Functional1) with <img src="https://render.githubusercontent.com/render/math?math=I(x %2B d_x(x,y),y %2B d_y(x,t),t %2B 1)-I(x,y,t)">, we have the following modified version of the functional to be minimized:

<p align="center">
  <img src="equation_26.png" width="500" id="L1Functional2">     [26]
</p>

with

<p align="center">
  <img src="equation_27.png" width="500" id="xxx">     [27]
</p>


The term <img src="https://render.githubusercontent.com/render/math?math=I(x %2B d_x(x,y),y %2B d_y(x,y),t %2B 1)"> is non-linear in <img src="https://render.githubusercontent.com/render/math?math=(d_x,d_y)"> and is typically linearized using Taylor expansion leading to

<p align="center">
  <img src="equation_28.png" width="900" id="rhoFunctionTVL1">     [28]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=(d_x^0(x,y),d_y^0(x,y))"> is an approximation to <img src="https://render.githubusercontent.com/render/math?math=(d_x(x,y),d_y(x,y))">.  
Nevertheless, to allow discontinuities in the flow field, references propose to minimize the following functional:

<p align="center">
  <img src="equation_29.png" width="500" id="L1Functional3">     [29]
</p>

As it can be seen, the new functional does not contain squares and, as a matter of fact, contains a <img src="https://render.githubusercontent.com/render/math?math=L^1"> data penalty term. A penalizationlike this is known as *total variation regularization*. 
A way to minimize functional [\[29\]](#L1Functional3) is to introduce the following convex relaxation problem

<p align="center">
  <img src="equation_30.png" width="700" id="L1FunctionalRelaxation">     [30]
</p>

Such functional can be optimized by performing alternate minimizations against <img src="https://render.githubusercontent.com/render/math?math=(d_x,d_y)"> for a fixed <img src="https://render.githubusercontent.com/render/math?math=(v_x,v_y)"> and against <img src="https://render.githubusercontent.com/render/math?math=(v_x,v_y)"> for a fixed <img src="https://render.githubusercontent.com/render/math?math=(d_x,d_y)">. In other words,

  - for fixed <img src="https://render.githubusercontent.com/render/math?math=(v_x,v_y)">, 
    <p align="center">
        <img src="equation_31.png" width="400" id="func1TVL1">     [31]
    </p>
    is minimized;

  - for fixed <img src="https://render.githubusercontent.com/render/math?math=(d_x,d_y)">, 
    <p align="center">
        <img src="equation_32.png" width="400" id="func1TVL2">     [32]
    </p>
    is minimized.

Concerning the minimization of functionals [\[31\]](#func1TVL1) and [\[32\]](#func1TVL2):

  - functional [\[31\]](#func1TVL1) is minimized by using the following iterations
    <p align="center">
        <img src="equation_33.png" width="300" id="timeStepTVL1">     [33]
    </p>
    
    where <img src="https://render.githubusercontent.com/render/math?math=\tau"> is the time step, and performing the unknown update
    as
    <p align="center">
        <img src="equation_34.png" width="300" id="xxx">     [34]
    </p>

  - functional [\[32\]](#func1TVL2) is minimized using the following iterations
    <p align="center">
        <img src="equation_35.png" width="300" id="xxx">     [35]
    </p>
    
    where <img src="https://render.githubusercontent.com/render/math?math=TH(\cdot,\cdot)"> is a thresholding operator.

The above alternate minimization proceeds by the *warping* technique. In other words, the alternate minimization is repeated a certain number of times and, each time it is completed, the value of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}^0"> is updated and, consequently, also the values of <img src="https://render.githubusercontent.com/render/math?math=(I(x %2B d_x^0(x,y),y %2B d_y^0(x,y),t %2B 1)"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\nabla}I(x %2B d_x^0(x,y),y %2B d_y^0(x,y),t %2B 1)"> are updated. The number of warpings is a parameter that assures the stability of the method. It affects the running time and establishes a compromise between speed and accuracy.  
The iterations stop when the norm of the difference between <img src="https://render.githubusercontent.com/render/math?math=(d_x(x,y),d_y(x,y))"> at the current iteration and
<img src="https://render.githubusercontent.com/render/math?math=(d_x(x,y),d_y(x,y))"> at the previous iteration is less than a constant <img src="https://render.githubusercontent.com/render/math?math=\epsilon">. <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is a trade-off between precision and running time. A small value yields more accurate solutions at the expense of a slower convergence.  
Functional [\[29\]](#L1Functional3) can be rewritten by adding a third component to the <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}=(d_x,d_y)"> vector, namely, regarding it as <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}=(d_x,d_y,d_z)"> and regarding the images as functions of <img src="https://render.githubusercontent.com/render/math?math=(x,y,z)"> instead of <img src="https://render.githubusercontent.com/render/math?math=(x,y)">. This requires, for the evaluation of the gradient in [\[28\]](#rhoFunctionTVL1), the definition of the derivative of <img src="https://render.githubusercontent.com/render/math?math=I"> with respect to the new coordinate <img src="https://render.githubusercontent.com/render/math?math=z">. Such a derivative is conventionally set to <img src="https://render.githubusercontent.com/render/math?math=\gamma">. The added third dimension is conceived as a brightness variation term. In this way, an extra degree of freedom that allows for brightness changes that are not due to image movement is allowed. The inclusion of these brightness changes in the smoothness term, namely
<img src="https://render.githubusercontent.com/render/math?math=|\mathbf{\nabla}d_x| %2B |\mathbf{\nabla}d_y| %2B |\mathbf{\nabla}d_z|">, penalises high frequency brightness changes between frames.

## Theory: Dense and sparse optical flows using Lucas-Kanade’s approach

The Lucas-Kanade algorithm  was originally proposed in 1981 for dense processing. However, it can be easily applied in a sparse context since, coherently with what above said, it uses only local image information through the exploitation of a small window surrounding each point of interest. Accordingly, it has become also an important sparse technique. In the examples that will follow, we will see Lucas-Kanade’s approach at work both for dense and sparse applications.  
Lucas-Kanade’s approach relies on equation [\[6\]](#normalFlow). However, as we have already said, being dealt with pixelwise, such equation has two unknowns and consequently it is not uniquely solvable. For this reason, the approach uses a spatial coherence assumption to solve this problem, namely, neighboring points in a scene belong to the same surface, have similar motion, and project to nearby points on the image plane.  
In other words, by considering a pixel window around a certain pixel of interest, equation [\[6\]](#normalFlow) is transformed to

<p align="center">
    <img src="equation_36.png" width="300" id="lucasKanade">     [36]
</p>
    
where <img src="https://render.githubusercontent.com/render/math?math=I_x"> and <img src="https://render.githubusercontent.com/render/math?math=I_y"> are the spatial derivatives of <img src="https://render.githubusercontent.com/render/math?math=I"> along and <img src="https://render.githubusercontent.com/render/math?math=x"> and <img src="https://render.githubusercontent.com/render/math?math=y">, respectively, <img src="https://render.githubusercontent.com/render/math?math=I_t"> is its temporal derivative and <img src="https://render.githubusercontent.com/render/math?math=(m_1,n_1),\ldots,(m_{25}, n_{25})"> are the pixels inside the <img src="https://render.githubusercontent.com/render/math?math=5\times 5"> window. Equation [\[36\]](#lucasKanade) expresses the fact that the <img src="https://render.githubusercontent.com/render/math?math=x"> and <img src="https://render.githubusercontent.com/render/math?math=y"> components of the pixel velocity are assumed to be constant within the considered <img src="https://render.githubusercontent.com/render/math?math=5\times 5"> window. While equation [\[6\]](#normalFlow) was an
undetermined system of one equation in two unknowns, now equation [\[36\]](#lucasKanade) is, in our example, an overconstrained system of in two unknowns. A least-squares solution to it can be obtained as the minimum of the functional:

<p align="center">
    <img src="equation_37.png" width="150" id="lucasKanadeFunctional">     [37]
</p>

The solution to [\[37\]](#lucasKanadeFunctional) can be written as:

<p align="center">
    <img src="equation_38.png" width="150" id="xxx">     [38]
</p>

Obviously, to define a meaningful solution, <img src="https://render.githubusercontent.com/render/math?math=\left(\mathbf{A}^t\mathbf{A}\right)"> must be invertible and this typically occurs at the corners of the image. We will discuss below how spotting the image positions where <img src="https://render.githubusercontent.com/render/math?math=\left(\mathbf{A}^t\mathbf{A}\right)"> exhibits the best properties to be invertible and then where applying the Lucas-Kanade algorithm.  
An issue arising in the previously described approaches, both for dense and sparse applications, is the difficulty in following a fast point motion from an image to the next. In the following section, we will illustrate a strategy, namely, the *pyramids approach*, capable to mitigate such a limitation.

## The pyramids approach

The up to here recalled schemes make an explicit or implicit use of window functions to isolate the neighborhood of the point around which we want to compute the displacement. We indeed recall that Farneb<span>ä</span>ck’s approach has been presented using a window function <img src="https://render.githubusercontent.com/render/math?math=w"> while Brox *et al.*’s approach, the approach using TV-L1 regularization and the Lucas-Kanade’s method use, as a matter of fact, rectangular windows. In principle, the windows should have very small support so to favor resolution. Unfortunately, the disadvantage of using small local windows in the above-described methods is that large motions can move points outside of the local window when passing from an image to the next. In such a case, recovering the motion would be impossible. This problem led to development of “pyramidal” algorithms. Such algorithms perform the tracking starting from the top of the pyramid and using a window with large support leading to coarse resolution. Then,
they improve the resolution by traversing the pyramid to its bottom and using even narrower windows. Tracking over image pyramids then allows large motions to be caught by local windows.  
Let us now attempt to better explain the above exposed concepts.  
An important assumption of the optical flow algorithms we have presented is the *temporal persistence* or *small movements* assumption. In other words, the temporal increments from frame to frame are small enough that the objects do not move much from one frame to the next. Another important assumption is the *spatial coherence* one. According to it,
neighboring points in a scene belong to the same surface, have similar motion, and project to nearby points on the image plane. However, sometimes, due to camera limitations, objects move at high speed and, to catch them across frames, we need large windows. Unfortunately, this may break the spatial coherence assumption.  
To counteract this issue, the pyramids approach we will now going to sketch is used. Using pyramids, a first approximation of large motions is estimated by the same involved images, but with reduced resolution. This rough estimation is refined, in subsequent steps, by increasing the images resolution from coarse to high. This not only improves the
accuracy, but reduces also the computational complexity.  
To illustrate the pyramids approach, let <img src="https://render.githubusercontent.com/render/math?math=G(m,n)"> and <img src="https://render.githubusercontent.com/render/math?math=H(m,n)"> be two two-dimensional, <img src="https://render.githubusercontent.com/render/math?math=M\times N"> sized grey scaled images defined, for simplicity, over a continuous two-dimensional domain of <img src="https://render.githubusercontent.com/render/math?math=(m,n)">
points, with <img src="https://render.githubusercontent.com/render/math?math=m=0,\ldots,M-1"> and <img src="https://render.githubusercontent.com/render/math?math=n=0,\ldots,N-1">.  
Let us now consider an image point <img src="https://render.githubusercontent.com/render/math?math=(m,n)"> of <img src="https://render.githubusercontent.com/render/math?math=G">. The goal of the tracking is to find the location <img src="https://render.githubusercontent.com/render/math?math=(m,n)=(m %2B d_m,n %2B d_n)"> on  <img src="https://render.githubusercontent.com/render/math?math=H"> such that <img src="https://render.githubusercontent.com/render/math?math=G(m,n)"> and <img src="https://render.githubusercontent.com/render/math?math=H(m %2B d_m,n %2B d_n)"> are “similar”. The vector <img src="https://render.githubusercontent.com/render/math?math=(d_m,d_n)"> is thus a measure of the velocity of a point at <img src="https://render.githubusercontent.com/render/math?math=(m,n)">, namely, of the optical flow at <img src="https://render.githubusercontent.com/render/math?math=(m,n)">. The notion of similarity is to be understood in a two-dimensional neighborhood sense which, in turn,
requires the introduction of a proper integration window. If <img src="https://render.githubusercontent.com/render/math?math=B_m"> and <img src="https://render.githubusercontent.com/render/math?math=B_n"> are two integers defining a window size, then <img src="https://render.githubusercontent.com/render/math?math=(d_m,d_n)"> can be defined as the vector that minimizes the residual function

<p align="center">
    <img src="equation_39.png" width="400" id="xxx">     [39]
</p>

In other words, the similarity at hand is measured using an image neighborhood of size <img src="https://render.githubusercontent.com/render/math?math=(2B_m %2B 1)\times (2B_n %2B 1)">. Typical values for <img src="https://render.githubusercontent.com/render/math?math=B_m"> and <img src="https://render.githubusercontent.com/render/math?math=B_n"> are from <img src="https://render.githubusercontent.com/render/math?math=2"> to <img src="https://render.githubusercontent.com/render/math?math=7"> pixels.  
Intuitively, and as above underlined, the integration window size, defined by <img src="https://render.githubusercontent.com/render/math?math=B_m"> and <img src="https://render.githubusercontent.com/render/math?math=B_n">, should be small not to “smooth out” the image details. Opposite to that, it should be large, as above mentioned, to handle large motions. The trade-off between small and large window sizes can be dealt with by using a pyramidal approach.  
Let us then now define the pyramid representation of <img src="https://render.githubusercontent.com/render/math?math=G">. Let <img src="https://render.githubusercontent.com/render/math?math=G^{(0)}=G"> be the “zero-th” level image, namely, the raw image or the highest resolution image. In a word, <img src="https://render.githubusercontent.com/render/math?math=G"> itself. <img src="https://render.githubusercontent.com/render/math?math=G^{(0)}"> of course has size <img src="https://render.githubusercontent.com/render/math?math=M^{(0)}\times N^{(0)}=M\times N">. The pyramid technique proceeds iteratively: compute <img src="https://render.githubusercontent.com/render/math?math=G^{(1)}"> from <img src="https://render.githubusercontent.com/render/math?math=G^{(0)}">, then compute
<img src="https://render.githubusercontent.com/render/math?math=G^{(2)}"> from <img src="https://render.githubusercontent.com/render/math?math=G^{(1)}">, etc. Generally speaking, if <img src="https://render.githubusercontent.com/render/math?math=1"> is a generic pyramidal level, then <img src="https://render.githubusercontent.com/render/math?math=G^{(1)}"> is the <img src="https://render.githubusercontent.com/render/math?math=M^{(l)}\times N^{(l)}"> image at level <img src="https://render.githubusercontent.com/render/math?math=l"> which is obtained by <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}"> as

<p align="center">
    <img src="equation_40a.png" width="300" id="xxx">
</p>
<p align="center">
    <img src="equation_40b.png" width="700" id="xxx">
</p>
<p align="center">
    <img src="equation_40.png" width="700" id="pyramidsEquation">     [40]
</p>

The pyramids approach is illustrated in the next figure [10](#pyramidsApproach).

<p align="center">
  <img src="pyramidsApproach.png" width="400" id="pyramidsApproach">
  <br>
     <em>Figure 10. The pyramids approach.</em>
</p>

Notice that the transformation from <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}"> to <img src="https://render.githubusercontent.com/render/math?math=G^{(l)}"> is a low-pass filtering which leads to a smoothing of the image at the upper level.  
Notice also that, in equation [\[40\]](#pyramidsEquation), the image at level <img src="https://render.githubusercontent.com/render/math?math=l"> is formed from the image at level <img src="https://render.githubusercontent.com/render/math?math=l-1"> by accessing <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}"> essentially with the index <img src="https://render.githubusercontent.com/render/math?math=(2m,2n)">, apart from <img src="https://render.githubusercontent.com/render/math?math=\pm 1"> shifts. Therefore, being <img src="https://render.githubusercontent.com/render/math?math=M^{(l-1)}\times N^{(l-1)}"> the image dimensions of <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}">, the following conditions must be met:

<p align="center">
    <img src="equation_41.png" width="200" id="pyramidsConditions">     [41]
</p>

However, the <img src="https://render.githubusercontent.com/render/math?math=\pm 1"> requires to break through the actual size of <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}">, so that proper boundary conditions must be enforced. A typical possibility is:

<p align="center">
    <img src="equation_42.png" width="300" id="xxx">     [42]
</p>

On substituting the maximum values of the <img src="https://render.githubusercontent.com/render/math?math=m"> and <img src="https://render.githubusercontent.com/render/math?math=n"> coordinates for <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}"> in equation [\[41\]](#pyramidsConditions), then we have:

<p align="center">
    <img src="equation_43.png" width="200" id="xxx">     [43]
</p>

which constraints the maximum dimensions of <img src="https://render.githubusercontent.com/render/math?math=G^{(l)}"> as a function of the maximum dimensions of <img src="https://render.githubusercontent.com/render/math?math=G^{(l-1)}">.  
The maximum level <img src="https://render.githubusercontent.com/render/math?math=L"> is the pyramid height and is chosen with a degree of arbitrarity. Typical values are <img src="https://render.githubusercontent.com/render/math?math=2">, <img src="https://render.githubusercontent.com/render/math?math=3"> and <img src="https://render.githubusercontent.com/render/math?math=4"> and, for practical image dimensions, there is no point into having <img src="https://render.githubusercontent.com/render/math?math=L>4"> since, rising up the pyramid, the dimensions of the image are halved from level to level. Furthermore, since the main reason of using the pyramid technique is to be able to handle large pixel movements, and, in particular, larger than <img src="https://render.githubusercontent.com/render/math?math=(d_m,d_n)">, then <img src="https://render.githubusercontent.com/render/math?math=L"> should also be chosen according to the maximum expected image flow. Since our focus will be on OpenCV with CUDA implementation below, then we skip further details on this point.  
It is now time to discuss how the pyramid can be used within an optical flow algorithm. In particular, we describe the traversal of the pyramid from level <img src="https://render.githubusercontent.com/render/math?math=l"> to level <img src="https://render.githubusercontent.com/render/math?math=l-1">. On assuming that the processing at level <img src="https://render.githubusercontent.com/render/math?math=l"> has been completed, we have at our disposal an estimate <img src="https://render.githubusercontent.com/render/math?math=(d_m^{(l)},d_n^{(l)})"> of the optical flow. Such an estimate can be used as a starting guess of the optical flow at level <img src="https://render.githubusercontent.com/render/math?math=l-1">. More in detail, the starting guess at level <img src="https://render.githubusercontent.com/render/math?math=l-1"> is constructed as

<p align="center">
    <img src="equation_44.png" width="200" id="pyramidGuess">     [44]
</p>

In equation [\[44\]](#pyramidGuess), we have exploited the relation between the image indices of levels <img src="https://render.githubusercontent.com/render/math?math=l-1"> and <img src="https://render.githubusercontent.com/render/math?math=l">, while the subscript <img src="https://render.githubusercontent.com/render/math?math=g"> stands for *guess*. The new estimate <img src="https://render.githubusercontent.com/render/math?math=(d_{m}^{(l-1)},d_{n}^{(l-1)})"> is achieved by minimizing the following functional

<p align="center">
    <img src="equation_45.png" width="600" id="xxx">     [45]
</p>

against the variable <img src="https://render.githubusercontent.com/render/math?math=(d_{m_\delta}^{(l-1)},d_{n_\delta}^{(l-1)})"> for a fixed value of <img src="https://render.githubusercontent.com/render/math?math=(d_{m_g}^{(l-1)},d_{n_g}^{(l-1)})">. It should be noticed that the size <img src="https://render.githubusercontent.com/render/math?math=(2B_m %2B 1)\times(2B_n %2B 1)"> is independent on <img src="https://render.githubusercontent.com/render/math?math=l">.

## Corner finding

As already mentioned, dense optical flow techniques apply to all the pixels in an image, but sparse approaches apply only to pixels showing specific “features”. For this reason, when applying sparse methods, it is necessary in advance to look for specific features or patterns which can be easily tracked . But what are these features?  
In order to understand what are the so-called *good features to track*, let us consider figure [11](#featureToTrack) in which a large image formed by a simple green rectangle is shown.

<p align="center">
  <img src="featureToTrack.png" width="400" id="featureToTrack">
  <br>
     <em>Figure 11. Illustrating the features to track.</em>
</p>

Next to the rectangle, three possible features to track are depicted, namely, features <img src="https://render.githubusercontent.com/render/math?math=A">, <img src="https://render.githubusercontent.com/render/math?math=B"> and <img src="https://render.githubusercontent.com/render/math?math=C">.  
Let us now suppose that we are appointed to spot the position of box <img src="https://render.githubusercontent.com/render/math?math=A"> within the left image of figure [11](#featureToTrack). Box <img src="https://render.githubusercontent.com/render/math?math=A"> represents the first feature whose tracking feasibility we are going to investigate. It appears that we would be in a serious trouble in spotting the position of box <img src="https://render.githubusercontent.com/render/math?math=A"> since there would be infinite, better <img src="https://render.githubusercontent.com/render/math?math=\infty^2">, possible solutions obtainable by moving the grey frame of box <img src="https://render.githubusercontent.com/render/math?math=A"> to the left or to the right, up or down within the green patch of figure [11](#featureToTrack). This occurs because the gray frame of box <img src="https://render.githubusercontent.com/render/math?math=A"> would produce “no gradient” when moving within the large green patch. Producing no gradient gives then rise to tracking ambiguities.  
Let us now evaluate the possibility of tracking box <img src="https://render.githubusercontent.com/render/math?math=B"> representing an edge. In this case, the range of solutions reduces from <img src="https://render.githubusercontent.com/render/math?math=\infty^2"> to <img src="https://render.githubusercontent.com/render/math?math=\infty">  since an ambiguity is obtained only by moving the search box to the left or to the right. The blue frame around box <img src="https://render.githubusercontent.com/render/math?math=B"> has gradient moving up or down but, this notwithstanding, localization is still not uniquely possible.  
Finally, let us consider the tracking of box <img src="https://render.githubusercontent.com/render/math?math=C"> representing a corner. In this case, no ambiguity would appear and we would drop the solutions from <img src="https://render.githubusercontent.com/render/math?math=\infty"> to <img src="https://render.githubusercontent.com/render/math?math=1">. Indeed, there is only one position for the red frame to exactly represent box <img src="https://render.githubusercontent.com/render/math?math=B"> over the large green
patch. The red frame would show gradient either moving left-right or moving up-down. Corners are thus good features to track since they are uniquely spottable within an image. 
But, how do we find corners in an image? That is, how can we operate the so called *feature detection*?  
Intuitively, following what above illustrated with the aid of figure [11](#featureToTrack), we should look for the image regions which have maximum variation when moved, by a small amount, in all directions around.  
Once spotted the “good” features within the image, the last question to answer is: how can we find the same features in the next image? This task requires the capability of adequately describing the found features so that they can be spotted also in the next image, a step known as *feature description*.  
In the two next subsections, we recall two among the most successful corner detectors, namely, Harris and Shi-Tomasi corner detectors.

### Harris corner detector

One of the first corner detection algorithms has been Harris corner detector developed by Harris and Stephens in 1988.  
To illustrate it, let us consider, without loss of generality, a grey scale, two-dimensional image <img src="https://render.githubusercontent.com/render/math?math=I">. Moreover, let us suppose to divide the image in many small patches. Let us consider one of the various patches covering the image, say <img src="https://render.githubusercontent.com/render/math?math=W">, and let us consider also its version shifted of an offset <img src="https://render.githubusercontent.com/render/math?math=(d_x,d_y)">. Finally, let us consider the functional composed by the squared sums of the differences of the two
patches (original and shifted), namely

<p align="center">
    <img src="equation_46.png" width="300" id="functionalHarris">     [46]
</p>

In [\[46\]](#functionalHarris), the summation terms could in principle be weighted by the values of a proper window function, but this point has been here skipped for the sake of simplicity.  
For patches where the image is nearly constant, then the quantity <img src="https://render.githubusercontent.com/render/math?math=\left(I(x,y)-I(x %2B d_x,y %2B d_y)\right)"> is close to vanishing. Opposite to that, for patches where the image has significant changes and that potentially contain an edge, such difference is larger. Accordingly, and following the discussion involving figure [11](#featureToTrack), we need to search for the patches where <img src="https://render.githubusercontent.com/render/math?math=E(d_x,d_y)"> is large.  
To this end, on expanding the quantity <img src="https://render.githubusercontent.com/render/math?math=I(x %2B d_x,y %2B d_y)"> in a Taylor series for a small shift <img src="https://render.githubusercontent.com/render/math?math=(d_x,d_y)">, then we have

<p align="center">
    <img src="equation_47.png" width="300" id="harrisTaylor">     [47]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=I_x"> and <img src="https://render.githubusercontent.com/render/math?math=I_y"> are the partial derivatives of <img src="https://render.githubusercontent.com/render/math?math=I">. On substituting equation [\[47\]](#harrisTaylor) in
[\[46\]](#functionalHarris), we finally have

<p align="center">
    <img src="equation_48.png" width="300" id="functionalHarris2">     [48]
</p>

Functional [\[48\]](#functionalHarris2) can be conveniently rewritten as

<p align="center">
    <img src="equation_49.png" width="300" id="xxx">     [49]
</p>

where

<p align="center">
    <img src="equation_50.png" width="400" id="cornerMatrix">     [50]
</p>

The so-obtained matrix <img src="https://render.githubusercontent.com/render/math?math=\mathbf{M}"> is symmetric and positive, semi-definite, so that it can be decomposed by the SVD as

<p align="center">
    <img src="equation_51.png" width="200" id="xxx">     [51]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\sigma_1"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2"> are singular values. The SVD has been already used in [Face recognition using OpenCV and CUDA](https://vitalitylearning2021.github.io/faceRecognitionOpenCVCUDA/) to implement face recognition.  
The shift <img src="https://render.githubusercontent.com/render/math?math=[d_x d_y]^t"> can be then rewritten by the basis vectors forming the columns of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{U}">, namely, as

<p align="center">
    <img src="equation_52.png" width="300" id="harrisShiftExpansion">     [52]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2"> are the columns of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{U}"> and <img src="https://render.githubusercontent.com/render/math?math=a_1"> and <img src="https://render.githubusercontent.com/render/math?math=a_2"> are the components of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{d}"> along the basis offered by <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2">. Following equation [\[52\]](#harrisShiftExpansion), we have

<p align="center">
    <img src="equation_53.png" width="200" id="harrisShiftExpansionFinal">     [53]
</p>

According to the geometric interpretation of the SVD, equation [\[53\]](#harrisShiftExpansionFinal) says that:

1.  When <img src="https://render.githubusercontent.com/render/math?math=\sigma_1 \gg 0"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2\simeq 0">, then we have a strong variation of the image intensity along <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1"> and a small variation of the image intensity along <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2"> within the patch <img src="https://render.githubusercontent.com/render/math?math=W">. This means that <img src="https://render.githubusercontent.com/render/math?math=W"> contains an edge in the direction orthogonal to <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1">. Similarly, when <img src="https://render.githubusercontent.com/render/math?math=\sigma_1 \simeq 0"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2 \gg 0">, then <img src="https://render.githubusercontent.com/render/math?math=W"> contains an edge in the direction orthogonal to <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2">.
2.  When <img src="https://render.githubusercontent.com/render/math?math=\sigma_1 \gg 0"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2 \gg 0">, then within the patch <img src="https://render.githubusercontent.com/render/math?math=W"> we have a strong variation of the image intensity both along <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2">. This means that <img src="https://render.githubusercontent.com/render/math?math=W"> contains a corner.
3.  If <img src="https://render.githubusercontent.com/render/math?math=\sigma_1 \simeq 0"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2 \simeq 0">, then <img src="https://render.githubusercontent.com/render/math?math=W"> contains neither an edge nor a corner.

The computation of the SVD of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{M}"> should be performed for each patch in which the image is divided. This can be computationally burdened. Therefore, instead of computing the full SVD, the possibility of using a less burdened and faster to be computed indicator is explored. Typically, the following indicator can be used

<p align="center">
    <img src="equation_54.png" width="200" id="harrisParameter">     [54]
</p>

where <img src="https://render.githubusercontent.com/render/math?math=\det{\cdot}"> indicates the determinant of a matrix, <img src="https://render.githubusercontent.com/render/math?math=\mathrm{Tr}{\cdot}"> is its trace and <img src="https://render.githubusercontent.com/render/math?math=k"> is an empirically determined constant typically belonging to <img src="https://render.githubusercontent.com/render/math?math=[0.04, 0.06]">. The following equalities hold

<p align="center">
    <img src="equation_55.png" width="120" id="xxx">     [55]
</p>

and

<p align="center">
    <img src="equation_56.png" width="120" id="xxx">     [56]
</p>

Accordingly:

1.  If <img src="https://render.githubusercontent.com/render/math?math=|R|"> is small, which occurs when <img src="https://render.githubusercontent.com/render/math?math=\sigma_1"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2"> are small, the patch contains neither a corner, nor an edge.
2.  If <img src="https://render.githubusercontent.com/render/math?math=R<0">, which occurs when <img src="https://render.githubusercontent.com/render/math?math=\sigma_1 \gg \sigma_2"> or vice versa, the patch contains an edge.
3.  if <img src="https://render.githubusercontent.com/render/math?math=R"> is large, which happens when <img src="https://render.githubusercontent.com/render/math?math=\sigma_1"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2"> are large and <img src="https://render.githubusercontent.com/render/math?math=\sigma_1\simeq \sigma_2">, the patch contains a corner.

The three above possibilities are represented in a nice picture as follows:

<p align="center">
  <img src="harrisCorner.png" width="400" id="harrisCorner">
  <br>
     <em>Figure 12. Illustrating the Harris corner detector.</em>
</p>

In figure [12](#harrisCorner), the plane of the singular values is represented. For what above said, the top-left region spots an edge in the direction orthogonal to <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2">, while the bottom-right region an edge in the direction orthogonal to <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1">. The region around the origin is “flat” in the sense that it represents neither corners, nor edges. The portion of plane around the bisector corresponds to the case when both <img src="https://render.githubusercontent.com/render/math?math=\sigma_1"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2"> are significant and so the patch has variations both along <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_1"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{u}_2"> which corresponds to a corner.  
In next section, we will shortly describe the Shi-Tomasi corner detector, namely, another corner detector often used in applications and, in particular, in the applications of interest of this chapter.

### Shi-Tomasi corner detector

Shi and Tomasi corner detector  works along the same line as that for Harris corner detector seen above, but uses a modified indicator. Instead of exploiting [\[54\]](#harrisParameter), Shi and Tomasi proposed using

<p align="center">
    <img src="equation_57.png" width="150" id="xxx">     [57]
</p>

If <img src="https://render.githubusercontent.com/render/math?math=R"> is greater than a certain threshold, the patch is considered to host a corner. As for the Harris corner detector, also Shi and Tomasi corner detector can be illustrated by a graph. If we use again the <img src="https://render.githubusercontent.com/render/math?math=(\sigma_1,\sigma_2)"> plane as we have done for Harris corner detector, we get the image in the figure [13](#shiTomasiCorner):

<p align="center">
  <img src="shiTomasiCorner.png" width="400" id="shiTomasiCorner">
  <br>
     <em>Figure 13. Illustrating the Shi and Tomasi corner detector.</em>
</p>

From the above figure, it can be seen that only when <img src="https://render.githubusercontent.com/render/math?math=\sigma_1"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_2"> are above a minimum value <img src="https://render.githubusercontent.com/render/math?math=\sigma_{\min}">, the patch is considered hosting a corner. The corner region is the green region in figure [13](#shiTomasiCorner).  
Under certain assumptions, the corner detected by the Shi-Tomasi corner detector are more stable for tracking than the Harris one.

## Optical flow visualization

Once obtained the optical flow, which is a measure of the displacement of image points from an image to the next, it is necessary to proceed to its visualization. However, being the displacement a vector quantity, its visualization is not straightforward.  
In the following, we will see two different optical flow visualization schemes. The first refers to the dense case and uses a color coding of the determined vector quantity. The second regards the sparse case and uses a representation by “streamlines”.  
In the next subsection, we examine first the dense case.

### Color coding

As mentioned above, in the dense optical flow techniques, the result is a pixel-wise map of image velocity. The problem here is visualizing the result of the processing, namely, a two-dimensional map of a two-dimensional vector.  
A way to pursue this end is coding the two-dimensional vector with a color map in the Hue-Saturation-Value (HSV) format. While, in the Red-Green-Blue (RGB) format, each color is obtained as a combination of red, blue and green, the HSV format aims at representing the colors using the same attributes by which human vision perceives the chromatic
properties of objects. In particular, HSV uses a cylindrical geometry in which the value is the vertical axis and represents the mixture of paints with different quantities of black or white paint; the hue is the angular dimension, starting with the red, passing through the green and the blue and then wrapping back to red; the saturation is the radial dimension and accounts for various tints of brightly colored paint. Hue, value and saturation are illustrated in next figure [14](#HSV).

<p align="center">
  <img src="HSV.jpg" width="400" id="HSV">
  <br>
     <em>Figure 14. Illustrating the HSV color coding.</em>
</p>

Indeed, a visible light source does not typically emit purely monochromatic light, but light having a certain bandwidth corresponding to a certain range of wavelengths. Such a concept is illustrated in Fig. [15](#saturationPlot).

<p align="center">
  <img src="saturationPlot.jpg" width="400" id="saturationPlot">
  <br>
     <em>Figure 15. Three different spectra to illustrate color
coding.</em>
</p>

The above figure shows three possible spectra, all of them centered around the <img src="https://render.githubusercontent.com/render/math?math=500nm"> wavelength. However, the three spectra have different distributions: spectrum <img src="https://render.githubusercontent.com/render/math?math=A"> is much concentrated around <img src="https://render.githubusercontent.com/render/math?math=500nm">, spectrum <img src="https://render.githubusercontent.com/render/math?math=B"> is less concentrated around <img src="https://render.githubusercontent.com/render/math?math=500nm"> and spectrum <img src="https://render.githubusercontent.com/render/math?math=A"> is widespread.  
In the HSV language, *hue* is represented by the frequency, and so by the color, around which the spectrum is centered. Opposite to that, *saturation* is represented by the bandwidth. Spectrum <img src="https://render.githubusercontent.com/render/math?math=A"> of Fig. [15](#saturationPlot) has a high saturation level and the corresponding light appears mostly “pure”. Opposite to that, spectra <img src="https://render.githubusercontent.com/render/math?math=B"> and <img src="https://render.githubusercontent.com/render/math?math=C"> correspond to lower saturation levels and so their lights appear less pure. High saturation levels make the color appear purer, while lower saturation levels make the color appear more gray. The last parameter, the *value*, corresponds to color brightness. Higher values make the color appear whiter, while lower values make the color appear blacker. MS paint color picker illustrated in the below figure offers a good test bench to better understand HSV color coding.

<p align="center">
  <img src="colorPicker.png" width="400" id="colorPicker">
  <br>
     <em>Figure 16. MS Paint color picker.</em>
</p>

As it can be seen from Fig. [16](#colorPicker), MS Paint enables representing a color either by using classical RGB (Red - Green - Blue) coding, or HSV.  
The color wheel is constructed by associating the hue to the angular coordinate of a polar system and the saturation to the radial coordinate. Concerning the angular coordinate, the full circle is partitioned in a certain number of sectors associated to a certain number of color transitions. In the following, we will deal with the
Middlebury color coding based on the way the human eye perceives the different color shades. In particular, the color transitions are \(6\): red-yellow (RY), yellow-green (YG), green-cyan (GC), cyan-blue (CB), blue-magenta (BM), magenta-red (MR). However, it should be noticed that such color transitions do not contain the same number of shades. This is illustrated in table [1](#colorShades) which reports the color transitions along with the corresponding number of shades in the Middlebury approach.

<p align="center">
  <img src="Table_1.png" width="400" id="colorShades">
  <br>
     <em>Table 1. Middlebury color transitions.</em>
</p>

We will resume table [1](#colorShades) when we will discuss the implementations below.

<p align="center">
  <img src="colorWheel.jpg" width="400" id="xxx">
  <br>
     <em>Figure 17. Color wheel.</em>
</p>

It is now time to practice the theory, both for the dense and for the sparse optical flow cases. In next section, we will start practicing the dense case.

## Practice: dense optical flow

We turn now to the presentation of the code for the computation of dense optical flow.  
The first step is to choose two test image for the optical flow computation. The images that we will here consider for either dense or sparse optical flows have been drawn from the [Reinhard Klette’s page](https://ccv.wordpress.fos.auckland.ac.nz/data/stereo-pairs/) and are shown in the next two figures:

<p align="center">
  <img src="firstImage.jpg" width="400" id="firstImage">
  <br>
     <em>Figure 18. First image for dense optical flow calculation.</em>
</p>

<p align="center">
  <img src="secondImage.jpg" width="400" id="secondImage">
  <br>
     <em>Figure 19. Second image for dense optical flow calculation.</em>
</p>

As it can be seen from figures [18](#firstImage) and [19](#secondImage), the two images are very similar since they have been acquired at two very close instants of time. Accordingly, although they aim at representing movements across the scene, the movement itself is not really visible to the naked eye. The snipped defining the names
of the two images is the following:

``` c++
string filename1 = "./rect_0384_c1.tif";
string filename2 = "./rect_0385_c1.tif";
```

As first operation, the two images are loaded and stored in a `Mat` matrix, in grey scale format:

``` c++
Mat im0 = imread(filename1, IMREAD_GRAYSCALE);
Mat im1 = imread(filename2, IMREAD_GRAYSCALE);
```

Once loaded the images, the success of the loading operation is checked. In particular, it is verified that the two images have been successfully opened, namely, they are not empty and are of the same size. This occurs by the snippet:

``` c++
const int fileCheckInt = fileCheck(im0, im1, filename1, filename2);
if (fileCheckInt == -1) return -1;
```

stopping the execution in case of any issue and exploiting the function `fileCheck()` below reported:

``` c++
int fileCheck(Mat &im0, Mat &im1, string &filename1, string &filename2) {
    
    if (im0.empty()) {
        cerr << "Image file [" << filename1 << "] can't be opened. Please, check." << endl;
        return -1; }
    
    if (im1.empty()) {
        cerr << "Image file [" << filename2 << "] can't be opened. Please, check." << endl;
        return -1; }

    if (im1.size() != im0.size()) {
        cerr << "Images are not of equal size. Please, check." << endl;
        return -1; }
}
```
<p align="center" id="xxx" >
     <em>Listing 1. The `fileCheck()` function.</em>
</p>

Once verified that the loading has been successful, the images are transferred from host to device:

``` c++
GpuMat d_im0(im0);
GpuMat d_im1(im1);
```

and the `d_opticalFlow` GPU matrix is defined. Matrix `d_opticalFlow` is responsible to store the result of the optical flow calculation and has the same dimensions of the loaded test images.  
Later on, the color wheel is computed. This occurs thanks to the following function:

``` c++
void computeColorWheel() {

    int k = 0;
    for (int i = 0; i < RY; ++i, ++k) colorWheel[k] = 
        Vec3i(255, 255 * i / RY, 0);
    for (int i = 0; i < YG; ++i, ++k) colorWheel[k] = 
        Vec3i(255 - 255 * i / YG, 255, 0);
    for (int i = 0; i < GC; ++i, ++k) colorWheel[k] = 
        Vec3i(0, 255, 255 * i / GC);
    for (int i = 0; i < CB; ++i, ++k) colorWheel[k] = 
        Vec3i(0, 255 - 255 * i / CB, 255);
    for (int i = 0; i < BM; ++i, ++k) colorWheel[k] = 
        Vec3i(255 * i / BM, 0, 255);
    for (int i = 0; i < MR; ++i, ++k) colorWheel[k] = 
        Vec3i(255, 0, 255 - 255 * i / MR); }
```
<p align="center" id="xxx" >
     <em>Listing 2. The `computeColorWheel()` function.</em>
</p>

The `computeColorWheel()` function defines the colour shades in the different transitions described by table [1](#colorShades).  
Performed such preparatory operations, it is time to proceed to the actual optical flow computation by using the above described approaches.

### Practice: Dense optical flow using Farneb<span>ä</span>ck’s approach

The first considered method is Farneb<span>ä</span>ck’s approach.  
The first operation is that of defining an object of the `cuda::FarnebackOpticalFlow` class. This occurs using the `cuda::FarnebackOpticalFlow::create()` function whose syntax is

``` c++
cuda::FarnebackOpticalFlow::create (int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13, int numIters=10, int polyN=5, double polySigma=1.1, int flags=0)
```

In the following, we report the detailed description of its input parameters, also according to what illustrated in section [5](#farnebackSection). In the presented algorith, the window <img src="https://render.githubusercontent.com/render/math?math=w"> in [\[functionalFarneback\]](#functionalFarneback) is assumed to be a box filter or a Gaussian bell.

1.  `numLevels` is the number of pyramid layers including the initial image; `numLevels=1` means that the pyramids approach is dismissed;
2.  `pyrScale` is the pyramid scale factor belonging to <img src="https://render.githubusercontent.com/render/math?math=(0,1)">; instead of using the standard downsampling factor of <img src="https://render.githubusercontent.com/render/math?math=0.5"> on each level which makes each next layer twice smaller than the previous one, an arbitrary factor is used;
3.  `fastPyramids` skips too small pyramid levels by cropping the pyramid levels which are smaller than `32 x 32` ;
4.  `winSize` is the averaging window size, namely, the support of the window <img src="https://render.githubusercontent.com/render/math?math=w"> used in functional
    [\[13\]](#functionalFarneback);
5.  `numIters` is the number of iterations the algorithm uses to minimize functional [\[13\]](#functionalFarneback) at each pyramid level;
6.  `polyN` is the size of the pixel neighborhood used to find the polynomial expansion [\[7\]](#farnebackApproximation) in each pixel; with larger `polyN`, the image will be approximated with smoother surfaces, resulting in a more robust algorithm, but with more blurred motion field; typically, `polyN=5` or `polyN=7`;
7.  `polySigma` is standard deviation of the Gaussian window in case a Gaussian window <img src="https://render.githubusercontent.com/render/math?math=w"> is used in [\[13\]](#functionalFarneback); for `polyN=5`, a good value is `polySigma=1.1`; for `polyN=7`, a good value is `polySigma=1.5`;
8.  `flags` can be a combination of the following:
    `OPTFLOW_USE_INITIAL_FLOW = 4` to use the input flow as an initial flow approximation; 
    `OPTFLOW_FARNEBACK_GAUSSIAN = 256` to use a Gaussian window <img src="https://render.githubusercontent.com/render/math?math=w"> instead of a box filter; usually, this option leads to better accuracy than a box filter, at the cost of lower speed; normally, `winSize` for a Gaussian window should be set to a larger value to achieve the same level of robustness.

Once the object of the class `cuda::FarnebackOpticalFlow` has been created, it is necessary to work out the optical flow calculations. This is performed by using the following snippet:

``` c++
farn->calc(d_im0, d_im1, d_opticalFlow);
```

The output matrix, namely, the `d_opticalFlow` GPU matrix, is a `float`, two-channels matrix. 
Finally, it is necessary to visualize the optical flow. This occurs by exploiting the `computeFlowAndShow()` function reported in the following

``` c++
static void computeFlowAndShow(const char *name, const GpuMat &d_opticalFlow) {
    // --- Split the x and y components of the displacement of a two channel matrix into an array of matrices
    GpuMat planes[2];
    cuda::split(d_opticalFlow, planes);

    // --- Copy the x and y flow components to CPU
    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    colorOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out); }
```
<p align="center" id="xxx" >
     <em>Listing 3. The `computeFlowAndShow()` function.</em>
</p>

The first operation executed by `computeFlowAndShow()` is splitting the two-channels matrix `d_opticalFlow` into the `planes` two matrices `GpuMat` array. This is possible thanks to

``` c++
cuda::split(d_opticalFlow, planes);
```

Later on, `planes[0]` and `planes[1]` are transferred to the two CPU `Mat` matrices `opticalFlowx` and `opticalFlowy`. Indeed, for the sake of simplicity, the association between optical flow and colours occurs on the host. The `computeFlowAndShow()` function, in turn, recalls the `colorOpticalFlow()` function. Such a function is reported in the code snippet below:

``` c++
static void colorOpticalFlow(const Mat_<float> &h_dx, const Mat_<float> &h_dy, Mat &h_coloredFlow, float maxmotion) {
    // --- Creates a colored flow of the same size of the flow and sets it to all zeros
    h_coloredFlow.create(h_dx.size(), CV_8UC3);
    h_coloredFlow.setTo(Scalar::all(0));

    for (int y = 0; y < h_dx.rows; ++y) {
        for (int x = 0; x < h_dx.cols; ++x) {
            
            Point2f u(h_dx(y, x), h_dy(y, x));

            // --- In the flow is valid, returns the color associated to the displacement
            if (isFlowValid(u)) h_coloredFlow.at<Vec3b>(y, x) = returnCOLOR(u.x / maxmotion, u.y / maxmotion); }}}
```
<p align="center" id="xxx" >
     <em>Listing 4. The `colorOpticalFlow()` function.</em>
</p>

It is appointed to define and initialize to zero the three-channels, `unsigned` matrix `h_coloredFlow`. Subsequently, each point of `h_coloredFlow` is coloured as a function of the two optical flow components `h_dx` and `h_dy`, after having performed a check, using the `isFlowValid` function, that the optical flow is valid. The optical flow is considered to be valid only if it is not `NaN` and if its components are not exceedingly large. The `colorOpticalFlow()` function colours the optical flow by exploiting the `returnCOLOR()` function reported in the following:

``` c++
static Vec3b returnCOLOR(float ux, float uy) {

    // --- Displacement: distance from origin
    const float rad             = sqrt(ux * ux + uy * uy);
    // --- Displacement: angular position normalized by pi (angular position comprised in (-1, 1)
    const float angleNormalized = atan2(-uy, -ux) / (float)CV_PI;

    // --- Divides the full circle into NSHADES slices and computes the position of the displacement vector
    //     within such slices
    const float shadingPosition = (angleNormalized + 1.0f) / 2.0f * (NSHADES - 1);
    // --- Beginning slice index
    const int   shade0  = static_cast<int>(shadingPosition);
    // --- Ending slice index
    const int   shade1  = (shade0 + 1) % NSHADES;
    // --- Offset wrt beginning slice index
    const float f   = shadingPosition - shade0;

    Vec3b displacementColor;

    for (int RGBcol = 0; RGBcol < 3; RGBcol++)
    {
        // --- Computes normalized RGB color corresponding to initial shade
        const float col0 = colorWheel[shade0][RGBcol] / 255.0f;
        // --- Computes normalized RGB color corresponding to final shade
        const float col1 = colorWheel[shade1][RGBcol] / 255.0f;

        // --- Computes hue of HSV according to angular position of displacement
        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            // --- Change saturation with radius. For rad = 0, col = 1; for rad = 1, col = col.
            col = 1 - rad * (1 - col); 
        else
            // --- Radius out of range
            col *= .75; 

        displacementColor[RGBcol] = static_cast<uchar>(255.0 * col);
    }

    return displacementColor; }
```
<p align="center" id="xxx" >
     <em>Listing 5. The `returnCOLOR()` function.</em>
</p>

The `returnCOLOR()` function first converts the Cartesian components of the optica flow in polar ones, namely, `rad` and `angleNormalized`, where `angleNormalized` has been normalized to <img src="https://render.githubusercontent.com/render/math?math=\pi">. Through `shadingPosition`, the colour wheel shading corresponding to the optical flow angle is evaluated. Later on, `shade0` and `shade1` compute the boundaries of the shading interval, while `f` computes the offset with respect to the left border of the shading interval itself. For each shading interval boundary, the corresponding colour, in RGB format, is computed leading to `col0` and `col1`. Finally, the value of `f` is used to perform a linear interpolation between `col0` and `col1`.  
The result of the whole processing can be appreciated from the following figure:

<p align="center">
  <img src="Farneback.JPG" width="400" id="Farneback">
  <br>
     <em>Figure 20. Result of the Farneb<span>ä</span>ck’s optical flow calculations.</em>
</p>

As it can be seen from figure [20](#Farneback), the coloured image regions are those relative to the pedestrian and to the moving cars. Such regions are differently coloured, being the corresponding scene movements different.
Let us now turn to the use of the Brox *et al.*’s approach.

### Practice: Dense optical flow using Brox *et al.*’s approach

Also in the case of the Brox *et al.*’s approach, the first operation is defining an object of the `cuda::BroxOpticalFlow` class. This takes place by using the `cuda::BroxOpticalFlow::create()` function whose prototype is:

``` c++
cuda::BroxOpticalFlow::create(double alpha=0.197, double gamma=50.0,
    double scale_factor=0.8, int inner_iterations=5, 
    int outer_iterations=150, int solver_iterations=10)
```

To reach a better understanding of such a function also in connection to what illustrated in section [6](#broxSection), in the following we report the detailed description of its input parameters:

1.  `alpha` is the regularization parameter in equation [\[18\]](#broxTwoTerms);
2.  `gamma` is the gradient constancy importance in equation [\[16\]](#broxFunctional1);
3.  `scale_factor` is the same as `pyrScale` for Farneb<span>ä</span>ck’s approach;
4.  `inner_iterations` is the number of inner loop iterations [\[21\]](#fixedPoint);
5.  `outer_iterations` is the number of pyramid levels;
6.  `solver_iterations` is number of linear system solver iterations.

Once defined the object of the `cuda::BroxOpticalFlow` class, the two images are first amplitude scaled so that their respective possible maxima are <img src="https://render.githubusercontent.com/render/math?math=1">:

``` c++
GpuMat d_im0Scaled;
GpuMat d_im1Scaled;

d_im0.convertTo(d_im0Scaled, CV_32F, 1.0 / 255.0);
d_im1.convertTo(d_im1Scaled, CV_32F, 1.0 / 255.0);
```

Subsequently, the very calculation of the optical flow occurs:

``` c++
brox->calc(d_im0Scaled, d_im1Scaled, d_opticalFlow);
```

Finally, the optical flow is visualized. The result of the processing according to Brox *et al.* is illustrated in the following figure:

<p align="center">
  <img src="Brox.JPG" width="400" id="Brox">
  <br>
     <em>Figure 21. Result of the Brox *et al.*’s optical flow calculations.</em>
</p>

As it can be seen from figure [21](#Brox), once again, the coloured image regions are those relative to the pedestrian and the moving cars.  
The next step for us is to experimenting dense optical flow using TV-L1 regularization.

### Practice: Dense optical flow using TV-L1 regularization

In a way totally analogous to the above examined techniques, the first step is using the `cuda::OpticalFlowDual_TVL1::create` function whose prototype is:

``` c++
cv::cuda::OpticalFlowDual_TVL1::create(double tau = 0.25, double lambda = 0.15, double theta = 0.3, int nscales = 5, int warps = 5, double epsilon = 0.01, int iterations = 300, double scaleStep = 0.8, double gamma = 0.0, bool  useInitialFlow = false)  
```

In the following, we report the detailed meaning of the input parameters of such a function:

1.  `tau` is the time step in equation [\[33\]](#timeStepTVL1);
2.  `lambda` is the weight in functional [\[29\]](#L1Functional3);
3.  `theta` is the convex relaxation parameter in [\[30\]](#L1FunctionalRelaxation);
4.  `nscales` is the number of pyramid levels;
5.  `warps` is the number of times that <img src="https://render.githubusercontent.com/render/math?math=I(x+d_x^0(x,y),y+d_y^0(x,y),t+1)"> and <img src="https://render.githubusercontent.com/render/math?math=\underline{\nabla}I(x+d_x^0(x,y),y+d_y^0(x,y),t+1)"> are computed per scale;
6.  `epsilon` stopping criterion threshold;
7.  `iterations` is the maximum number of iterations used for solving the convex relaxation problem at each pyramids scale and for each warping;
8.  `scaleStep` is the pyramids scale as before;
9.  `gamma` is the coefficient for additional, three-dimensional illumination variation term;
10. `useInitialFlow` establishes whether an initial flow guess should be used or not.

The actual computation is performed by the snippet

``` c++
tvl1->calc(d_im0, d_im1, d_opticalFlow);
```

The result is illustrated in the next figure [22](#TV-L1).

<p align="center">
  <img src="TVL1.JPG" width="400" id="TV-L1">
  <br>
     <em>Figure 22. Result of the TV-L1 optical flow calculations.</em>
</p>

Considerations totally analogous to the previous cases can be performed also for the current approach.  
In next section, we will deal with the last method considered for dense optical flow in this chapter, namely, Lucas-Kanade’s approach.

### Practice: Dense optical flow using Lucas-Kanade’s approach

In a way totally analogous to the previous cases, the definition of an object of the `cuda::DensePyrLKOpticalFlow` class is performed by the `cuda::DensePyrLKOpticalFlow::create()` function whose prototype is:

``` c++
cuda::DensePyrLKOpticalFlow::create(Size winSize = Size(13, 13), int maxLevel = 3, int iters = 30, bool useInitialFlow = false)
```

The input parameters of such a function have the following meaning:

1.  `winSize` is the size of the pixel window around the pixel of interest;
2.  `maxLevel` is the number of used pyramid levels; <img src="https://render.githubusercontent.com/render/math?math=0"> means no pyramids;
3.  `iters` is the number of iterations;
4.  `useInitialFlow` establishes whether an initial flow guess should be used or not.

Following the calculations performed invoking

``` c++
lk->calc(d_im0, d_im1, d_opticalFlow);
```

the result in figure [24](#LK) is obtained

<p align="center">
  <img src="LK.JPG" width="400" id="LK">
  <br>
     <em>Figure 23. Result of the Lucas-Kanade’s dense optical flow calculations.</em>
</p>

Next Listing reports the `main()` function containing all the illustrated approaches:

``` c++
int main() {
    
    // --- Images file names
    string filename1 = "./rect_0384_c1.tif";
    string filename2 = "./rect_0385_c1.tif";

    // --- Loading images into OpenCV matrices
    Mat im0 = imread(filename1, IMREAD_GRAYSCALE);
    Mat im1 = imread(filename2, IMREAD_GRAYSCALE);

    // --- Checking wether files can be opened or images have the same size
    const int fileCheckInt = fileCheck(im0, im1, filename1, filename2);
    if (fileCheckInt == -1) return -1;
         
    // --- Moving images from CPU to GPU
    GpuMat d_im0(im0);
    GpuMat d_im1(im1);

    GpuMat d_opticalFlow(im0.size(), CV_32FC2);

    // --- Compute color wheel
    computeColorWheel();

    /*************/
    /* FARNEBACK */
    /*************/
    Ptr<cuda::FarnebackOpticalFlow>     farn = cuda::FarnebackOpticalFlow::create(6, 0.4, false, 13, 15, 5, 1.1, 256);
    {
        farn->calc(d_im0, d_im1, d_opticalFlow);

        computeFlowAndShow("Farnebäck", d_opticalFlow);
    }

    /*********************/
    /* BROX OPTICAL FLOW */
    /*********************/
    Ptr<cuda::BroxOpticalFlow>          brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    {
        GpuMat d_im0Scaled;
        GpuMat d_im1Scaled;

        d_im0.convertTo(d_im0Scaled, CV_32F, 1.0 / 255.0);
        d_im1.convertTo(d_im1Scaled, CV_32F, 1.0 / 255.0);

        brox->calc(d_im0Scaled, d_im1Scaled, d_opticalFlow);

        computeFlowAndShow("Brox et al.", d_opticalFlow);
    }

    /**************/
    /* DUAL TV-L1 */
    /**************/
    Ptr<cuda::OpticalFlowDual_TVL1>     tvl1 = cuda::OpticalFlowDual_TVL1::create();
    {
        tvl1->calc(d_im0, d_im1, d_opticalFlow);

        computeFlowAndShow("TVL1", d_opticalFlow);
    }

    /****************/
    /* LUCAS-KANADE */
    /****************/
    Ptr<cuda::DensePyrLKOpticalFlow>    lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
    {
        lk->calc(d_im0, d_im1, d_opticalFlow);

        computeFlowAndShow("Lucas-Kanade", d_opticalFlow);
    }

    imshow("Frame 0", im0);
    imshow("Frame 1", im1);
    waitKey();

    return 0;
}
```
<p align="center" id="xxx" >
     <em>Listing 6. The `main()` function invoking dense optical flow functions.</em>
</p>

This was the last example of using OpenCV with CUDA to solve the dense optical flow problem. In next section, we will describe the considered algorithm to solve sparse optical flow using the CUDA-accelerated, OpenCV library.

## Practice: sparse optical flow

In this section, we will illustrate how, from a practical point of view, sparse optical flow in OpenCV with CUDA is performed. We will consider two cases, namely, the case when the optical flow is computed on the two test images previously considered for the dense application and the case when optical flow involves a sequence of video images.  
In next subsection, we will first consider the case of static images.

### Sparse optical flow on static images

Let us immediately get to the heart of the problem and illustrate the
sparse optical flow code.  
We will assume to operate on the same two test images used for the dense
case, so that the image loading does not need further commenting. We
just need to report that the two images are stored in the `im0` and
`im1` matrices.  
The first image is cast into grey scale and stored in the `im0Gray`
matrix. Indeed, the first step of the approach is that of determining
the good features to track and the relative CUDA-accelerated, OpenCV
routine requires grey scale images as inputs. The casting is worked out
using the function `cv::cvtColor()` as

``` c++
cv::cvtColor(im0, im0Gray, COLOR_BGR2GRAY);
```

As mentioned, the next step is that of determining the good features to
track. To this end, the first grey scale image is moved to the GPU by:

``` c++
GpuMat d_im0Gray(im0Gray);
```

The `d_previousPoints` `GpuMat` matrix is then declared. Such a matrix
will have the task of hosting the positions of the good features to
track in the first image. The name *previous* is related to the fact
that `d_previousPoints` hosts the mentioned positions with respect to
the first image and that these positions will be then tracked in the
second image. We will also deal, later on, with a `d_nextPoints`
`GpuMat`.  
The good features to track are determined using the function:

``` c++
cv::cuda::createGoodFeaturesToTrackDetector()
```

whose prototype is

``` c++
Ptr<CornersDetector> cv::cuda::createGoodFeaturesToTrackDetector
    (int srcType, int maxCorners = 1000, double qualityLevel = 0.01, 
    double minDistance = 0.0, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04)
```

The input parameters of such a routine have the following meaning:

1.  `srcType` is the input source type; only `CV_8UC1` and `CV_32FC1`
    are currently supported;

2.  `maxCorners` is the maximum number of good features to track; if
    more than `maxCorners` points are found, the strongest of them are
    returned;

3.  `qualityLevel` characterizes the minimal accepted quality of image
    corners; this parameter is multiplied by the minimum eigenvalue of
    matrix ([\[cornerMatrix\]](#cornerMatrix)) calculated across the
    image and the corners with the quality measure less than the product
    are rejected;

4.  `minDistance` is the minimum possible Euclidean distance between the
    returned corners;

5.  `blockSize` is the size of \(W\) in subsection [1.10.1](#Harris);

6.  `useHarrisDetector` parameter indicating whether to use Harris or
    Shi-Tomasi detectors;

7.  `harrisK` is the parameter \(k\) in equation
    ([\[harrisParameter\]](#harrisParameter)).

Once created an object of the class `CornersDetector` with the just
mentioned function, the actual computation is performed by:

``` c++
detector->detect(d_im0Gray, d_previousPoints);
```

After having computed and stored the points associated to the ai good
features to track in `d_previousPoints`, it is necessary to proceed to
the calculation of the optical flow associated to those points. This is
worked out by defining an object of the class
`cuda::SparsePyrLKOpticalFlow` using the
`cuda::SparsePyrLKOpticalFlow::create` function whose prototype is the
same of that for the already examined
`cuda::DensePyrLKOpticalFlow::create` function. The calculation is then
realized by the `SparseOpticalFlow::calc` function whose prototype is

``` c++
SparseOpticalFlow::calc(InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, OutputArray status, OutputArray err = cv::noArray())
```

The meaning of its input parameters is illustrated in the following:

1.  `prevImg` is the first input image;

2.  `nextImg` is the second input image;

3.  `prevPts` is the input vector of 2D points for which the flow is
    needed; in our case, it is the input vector of the points containing
    the good features to track;

4.  `nextPts` is the output vector of 2D points containing the computed
    new positions in the second image;

5.  `status` is the output status vector containing a `1` if the flow
    for the corresponding features has been found, otherwise a `0`;

6.  `err` is an optional output vector containing error response for
    each point (inverse confidence); the error is a measure of the
    similarity of the pixel values corresponding to the tracked features
    in the first image and the pixel values corresponding to the
    calculated optical flow in the second image.

The invokation of the `cuda::SparsePyrLKOpticalFlow::create` function is
the following:

``` c++
d_pyrLK_sparse->calc(d_frame0, d_frame1, d_previousPoints, d_nextPoints, d_status);
```

where `d_frame1`, `d_nextPoints` and `d_status` are `GpuMat` GPU
matrices. In particular, `d_frame1` is the GPU copy of the second image
`im1`.  
Once performed the computation, the results are copied from device to
host using the `device2Host` overloaded function reported below:

``` c++
static void device2Host(const GpuMat &d_A, vector<Point2f> &h_A) {
    h_A.resize(d_A.cols);
    Mat mat(1, d_A.cols, CV_32FC2, (void *)&h_A[0]);
    d_A.download(mat); }

static void device2Host(const GpuMat &d_A, vector<uchar> &h_A) {
    h_A.resize(d_A.cols);
    Mat mat(1, d_A.cols, CV_8UC1, (void *)&h_A[0]);
    d_A.download(mat); }
```

Finally, the results are visualized. This occurs thanks to the
`drawFlow()` function here reported:

``` c++
static void drawFlow(Mat &frame, const vector<Point2f> &previousPoints, const vector<Point2f> &nextPoints, const vector<uchar> &status, Scalar line_color = Scalar(0, 0, 255)) {

    // --- Loop over all the points
    for (size_t i = 0; i < previousPoints.size(); ++i) {
        
        // --- Check if point status is ok
        if (status[i]) {
            
            // --- Set line thickness
            int line_thickness = 1;

            // --- Set previous and next points
            Point p = previousPoints[i];
            Point q = nextPoints[i];

            // --- Find angle and length of arrow
            double angle = atan2((double)p.y - q.y, (double)p.x - q.x);
            double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + 
                                (double)(p.x - q.x)*(p.x - q.x));

            if (hypotenuse < 1.0) continue;

            // --- If the length of the arrow is less than 1, then lengthen the 
            //     arrow by a factor of three.
            q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
            q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

            // --- Define arrow line
            line(frame, p, q, line_color, line_thickness);

            // --- Draw the tips of the arrow. Some scaling is operated so that 
            //     the tips look proportional to the main line of the arrow.
            p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness); } } }
```

As it can be seen from Listing [\[drawFlow\]](#drawFlow), if the flow
for the corresponding features has been found (`status[i]==1`), the
`drawFlow()` function browses the good points to track evaluating their
positions in the first (`p`) and second (`q`) images. From such
information, the array joining them is computed, possibly magnifying it
if exceedingly short. The possibly magnified line joining `p` and `q` is
superimposed to the current image and drawn using the `line()` function.
It should be noticed that a solitary call to `line()` enables to draw
only a single line between `p` and `q`, so that other two invokations
are necessary to trace the array tips.  
The following Listing displays the `main()` function of the approach

``` c++
int main() {

    // --- Images file names
    string filename0 = "./rect_0384_c1.tif";
    string filename1 = "./rect_0385_c1.tif";

    Mat im0 = imread(filename0);
    Mat im1 = imread(filename1);

    cout << "Image size: " << im0.cols << " x " << im0.rows << endl;

    Mat im0Gray;

    // --- Converts images im0 and im1 to gray scale
    cv::cvtColor(im0, im0Gray, COLOR_BGR2GRAY);

    /***********************************/
    /* SPOTTING GOOD FEATURES TO TRACK */
    /***********************************/
    GpuMat d_im0Gray(im0Gray);
    GpuMat d_previousPoints;

    int     maxCorners = 4000;
    double  qualityLevel = 0.01;
    double  minDistance = 0;
    int     blockSize = 5;

    Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_im0Gray.type(), maxCorners, qualityLevel,
        minDistance, blockSize);
    detector->detect(d_im0Gray, d_previousPoints);

    /**************************************/
    /* LUCAS-KANADE'S SPARSE OPTICAL FLOW */
    /**************************************/
    GpuMat d_frame0(im0);
    GpuMat d_frame1(im1);
    GpuMat d_nextPoints;
    GpuMat d_status;

    int     winSize = 21;
    int     maxLevel = 3;
    int     iters = 30;
    Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(Size(winSize, winSize), maxLevel, iters);

    d_pyrLK_sparse->calc(d_frame0, d_frame1, d_previousPoints, d_nextPoints, d_status);

    // --- Copies from device to host
    vector<Point2f> h_previousPoints(d_previousPoints.cols);
    device2Host(d_previousPoints, h_previousPoints);

    vector<Point2f> h_nextPoints(d_nextPoints.cols);
    device2Host(d_nextPoints, h_nextPoints);

    vector<uchar> status(d_status.cols);
    device2Host(d_status, status);

    // --- Draw optical flow
    namedWindow("PyrLK [Sparse]", WINDOW_NORMAL);
    drawFlow(im0, h_previousPoints, h_nextPoints, status, Scalar(255, 0, 0));
    imshow("PyrLK [Sparse]", im0);

    waitKey(0);

    return 0; }
```

The result of the processing on the considered images is illustrated in
the following figure:

![Result of the Lucas-Kanade’s sparse optical flow calculations on
static images.](Pictures/Chapter04/sparseOpticalFlowPractice.JPG)

As it can be seen, the blue arrows indicate the movements of both the
pedestrian and the cars. In particular, they indicate that the
pedestrian and the car on the left move left, while the car on the right
moves right.  
In the following subsection, the sparse optical flow problem on video
images will be considered.

### Sparse optical flow on video images

We will here shortly describe the approach followed to compute the
sparse optical flow for video images, being that very similar to the one
already used for static images.  
The marked different with respect to the case examined in the foregoing
section is the use of the `VideoCapture` overloaded function. The
prototype of interest is the following:

``` c++
cv::VideoCapture::VideoCapture(const String &filename)  
```

Such a function opens a video file for video capturing. The first video
image is then stored as first image using the following command:

``` c++
cap >> im0;
```

The image in `im0` residing on the host is then copied to the device in
a `GpuMat` matrix `d_frame0`, scaled by `cuda::resize` and cast to grey
scale using `cuda::cvtColor`, namely:

``` c++
d_frame0.upload(im0);
cuda::resize(d_frame0, d_im0Gray, Size(d_frame0.cols * scale, d_frame0.rows * scale));
cuda::cvtColor(d_im0Gray, d_im0Gray, COLOR_BGR2GRAY);
```

Later on, as in the previous case, elements of the

``` c++
cuda::CornersDetector
cuda::SparsePyrLKOpticalFlow
```

classes are created. Moreover, an endless loop in which each new loaded
video image is stored in the “second” image `im1` is operated. `im1` is
then copied to the GPU in `d_frame1`, scaled and cast to grey scale. The
process then proceeds as in the foregoing case. At the end of each loop
cycle, `im1` is stored as “old” image in `im0`. The following Listing
reports the `main()` function of the approach:

``` c++
void main() {
    
    // --- CPU images
    Mat im0, im1;

    // --- GPU images
    cuda::GpuMat d_im0Gray, d_im1Gray, d_frame0, d_frame1;

    VideoCapture cap("Car - 16849.mp4");
    cap >> im0;
    if (im0.empty()) return;

    // --- Scaling factor
    //double scale = 800. / im0.cols;
    double scale = 1;

    // --- First image
    d_frame0.upload(im0);
    cuda::resize(d_frame0, d_im0Gray, Size(d_frame0.cols * scale, 
        d_frame0.rows * scale));
    cuda::cvtColor(d_im0Gray, d_im0Gray, COLOR_BGR2GRAY);

    /***********************************/
    /* SPOTTING GOOD FEATURES TO TRACK */
    /***********************************/
    int     maxCorners      = 4000;
    double  qualityLevel    = 0.01;
    double  minDistance     = 0;
    int     blockSize       = 5;
    Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_im0Gray.type(), maxCorners, qualityLevel, minDistance, blockSize);

    /**************************************/
    /* LUCAS-KANADE'S SPARSE OPTICAL FLOW */
    /**************************************/
    cuda::GpuMat d_previousPoints;
    cuda::GpuMat d_nextPoints;
    cuda::GpuMat d_status;

    int     winSize         = 21;
    int     maxLevel        = 3;
    int     iters           = 30;
    Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(winSize, winSize), maxLevel, iters);

    while (1) {

        // --- Get new image
        cap >> im1;
        if (im1.empty()) break;

        d_frame1.upload(im1);
        cuda::resize(d_frame1, d_im1Gray, Size(d_frame1.cols * scale, 
            d_frame1.rows * scale));
        d_im1Gray.download(im1);
        // --- Converts image to gray scale
        cuda::cvtColor(d_im1Gray, d_im1Gray, COLOR_BGR2GRAY);

        // --- Good features to track: detection
        detector->detect(d_im0Gray, d_previousPoints);
        
        // --- Compute optical flow
        d_pyrLK->calc(d_im0Gray, d_im1Gray, d_previousPoints, d_nextPoints, 
            d_status);

        // --- Swap old and new grey-scale images
        d_im0Gray = d_im1Gray;

        // --- Copies from device to host
        vector<Point2f> h_previousPoints(d_previousPoints.cols);
        device2Host(d_previousPoints, h_previousPoints);
        
        vector<Point2f> h_nextPoints(d_nextPoints.cols);
        device2Host(d_nextPoints, h_nextPoints);
        
        vector<uchar> status(d_status.cols);
        device2Host(d_status, status);
        
        // --- Draw optical flow
        drawFlow(im1, h_previousPoints, h_nextPoints, status, 
            Scalar(255, 0, 0));
        imshow("PyrLK [Sparse]", im1);

        if (waitKey(10) > 0) break; } }
```

The algorithm has been evaluated against a motorway traffic video
downloadable at
[pixabay](https://pixabay.com/videos/car-traffic-daytime-driving-on-road-16849/).
A frame of the processing results is reported in the below figure:

![Result of the Lucas-Kanade’s sparse optical flow calculations on video
images.](Pictures/Chapter04/LKVideo.jpg)

As it can be seen from the above image, all the good features to track
depart from the cars. Moreover, the arrows point in the correct
direction of travel and their lengths is proportional to the car speeds.
