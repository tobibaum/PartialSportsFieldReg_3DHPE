# Tutorial collection on how to generate synthetic datasets with groundtruth for 3D HPE research

In this tutorial, we approach the challenge of generating a synthetic dataset using Unreal Engine 5 to evaluate monocular 3D human pose estimation methods in a typical sports contexts. The two main challenges of this work are: 

(1) build a scene, create characters, animate the characters, and rig the camera path. 

(2) export the in-engine parameters into a txt file.

---
## 1. Building the scene
We grab assets from the following sources
- stadium from the UE [marketplace](https://www.unrealengine.com/marketplace/en-US/store)
- characters from [metahumans](https://metahuman.unrealengine.com/)
- animations from [mixamo](https://www.mixamo.com/#/)

With these, we create a scene in which the metahumans move according to mixamo motions. 

#### Prerequisites
The required steps to build such a scene have been elaborately described by game designers in many different ways. We used the following tutorials to learn about unreal engine without any prior knowledge:
- [YouTube: How to create a short film and set up camera](https://www.youtube.com/watch?v=eTWnzHQJvBE)
- [YouTube: Retarget metahuman to UE5](https://www.youtube.com/watch?v=4t61M-is_A8)
- [YouTube: Improve the quality of rendering output](https://www.youtube.com/watch?v=XgVOUCvrRW4&t=311)
- [YouTube: Add lens distortion](https://www.youtube.com/watch?v=aQESHDY7Aog)

#### Additional steps
*developed and tested in Unreal Engine 5.1.1*

Using the above tutorials, you should arrive at a sequence with a animated moving actor (`BP_ath0`) and a pan/tilt/zooming camera (`CineCameraActor`):
![unreal1](https://user-images.githubusercontent.com/1063330/230720007-bc835d3d-cd78-4a2c-b21a-80453dde397e.png)

You will need to add the following to the `Body`-portion of your athlete:
- `FKControlRig`: Right-click `Body` and select `Edit with FK Control Rig` which "bakes" in the motion sequence. You do not need to edit it in anyway after that.
- `PrintEvent`: click on `+ Track` next to `Body`, then `Event`->`Repeater` to create a hook that can be used inside the sequence's BluePrint. Double-click the resulting Event-track in the sequencer time view once to add it to the BluePrint.

Some steps required for the `CameraComponent` are:
- `PrintEvent`: click on `+ Track` next to `CameraComponent`, then `Event`->`Repeater` to create a hook that can be used inside the sequence's BluePrint. Double-click the resulting Event-track in the sequencer time view once to add it to the BluePrint.

Next you will have to add a point that the camera will focus on: 
- in the main interface, click on "Quickly add ..." (cube with green plus) -> Basic -> Actor, and create an actor `focusPoint`
- add `focusPoint` actor to the sequence
- animate `focusPoint`'s path to always be at the pelvis of th athlete
- in the "Details"-tab for the `CameraComponent`, find "Focus Settings", select focus method "tracking" and pick `focusPoint` as a track target.

## 2. Exporting the raw data
To store information to a file, install the [VictoryPlugin27](https://forums.unrealengine.com/t/39-ramas-extra-blueprint-nodes-for-you-as-a-plugin-no-c-required/3448).
Using this plugin, plus the event triggers described in the previous section, you can create blueprints to read and dump the respective groundtruth information to file:

#### Camera parameters
Straightforward grab and dump the parameters as shown in this blueprint:
![bp1b](https://user-images.githubusercontent.com/1063330/230721500-86da3cca-02d0-41d6-8bf7-0869d85dbced.png)

#### Joint positions
For the joints, you need to iterate over the parts of the body. this will contain way more points than necessary, which can be filtered later in python-land.
![bp3](https://user-images.githubusercontent.com/1063330/230721553-af4837ea-4c8c-470f-a8b0-622ae0fcd34e.png)

#### Projection matrices
You can also directly export projection matrices from unreal. (We did not use these in our work, since we did not manage to align them with our own projection matrix.)
![bp2](https://user-images.githubusercontent.com/1063330/230721773-c3ba6794-e1a5-4e9c-8f8c-d38f71237c3a.png)


#### Debugging
To debug the data and calculations in the previous steps, turn on the printing function in the blueprint (checkbox), and allow the EventTriggers to be executed during sequence testing: activate `Call in Editor`
![unreal2](https://user-images.githubusercontent.com/1063330/230721611-d8c407a7-b2fd-4b8b-a5b9-0a190f27fde9.png)


A complete `.ueasset` of the above steps for one of the presented sequences is available in [sciebo](https://dshs-koeln.sciebo.de/s/IIEsyX2gHRmgtZr), password: `baumgartner`
