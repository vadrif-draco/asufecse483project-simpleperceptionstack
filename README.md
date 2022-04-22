<!-- Much thanks to https://github.com/othneildrew/Best-README-Template for the template -->
<!-- And to https://github.com/alexandresanlim/Badges4-README.md-Profile for the badges -->
<img id="top" src="https://i.imgur.com/iW7JeHC.png" width="256" align="right" />

# Simple Perception Stack for Self-Driving Cars

[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
  
###### This is a _Simple Perception Stack for Self-Driving Cars_; the major task project for the **CSE483 - Computer Vision** course in the Faculty of Engineering, Ain Shams University; for Spring 2022.

<details>
  <summary><b>Table of Contents</b></summary>
	<ol>
		<li><a href="#foreword">Foreword</a></li>
    <li><a href="#phase-one-details-and-requirements">Phase One Details and Requirements</a></li>
    <li><a href="#phase-two-details-and-requirements">Phase Two Details and Requirements</a></li>
		<li><a href="#getting-started">Getting Started</a></li>
		<li><a href="#usage">Usage</a></li>
		<li><a href="#contributing">Contributing</a></li>
		<li><a href="#acknowledgments">Acknowledgments</a></li>
	</ol>
</details>

## Foreword
Self-driving cars have piqued human interest for centuries. Leonardo Da Vinci sketched out the plans for a hypothetical [self-propelled cart](https://en.wikipedia.org/wiki/Leonardo%27s_self-propelled_cart) in the late 1400s, and mechanical autopilots for airplanes emerged in the 1930s. In the 1960s an autonomous vehicle was developed as a possible moon rover for the Apollo astronauts. A true self-driving car has remained elusive until recently. Technological advancements in global positioning systems (GPS), digital mapping, computing power, and sensor systems have finally made it a reality.

In this project we are going to create a simple perception stack for self-driving cars (SDCs). Although a typical perception stack for a self-driving car may contain different data sources from different sensors (ex.: cameras, lidar, radar, etc…), we’re only going to be focusing on video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lanes and their lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks. The project is split into two phases.

###### Built With

[![Python][python-shield]][python-url]
[![OpenCV][opencv-shield]][opencv-url]
[![Numpy][numpy-shield]][numpy-url]
[![Jupyter Notebooks][jupyter-shield]][jupyter-url]
[![Google Colab][colab-shield]][colab-url]
<!-- [![Pandas][pandas-shield]][pandas-url] -->

## Phase One Details and Requirements

![Sample Before Computer Vision][before-vision]

The main perception stack, focusing on analyzing camera video streams caught by the vehicle itself from the front. This helps perform estimates on how to plan the next move, go left, go right, slow down, speed up, by how much, etc... Of course these estimates are not achievable through only analyzing the video stream, so this part is only concerned with what can be inferred from that simple means of information. That would be limited to detecting/identifying the boundaries of the lane that the car is currently driving on. Basically, the lane lines of the road ahead. This is under the assumption that road rules are being followed by drivers.

The lane detection will be manifested through coloring the borders (lines) of the lane with different colors as well as coloring the lane itself that lies in between. Additionally, some metrics describing the car's location and the lane's structure should be detected, such as the lane's radius of curvature and car's position w.r.t. the center of the lane. Of course, this will require conversion from pixels to meters. To simplify matters, the camera is assumed to be mounted at the car's center.

For consistency, the lane should be marked in green and its borders in yellow and/or red. The lanes should be detected in the pipeline as solid lines, interpolated to whatever function that fits them (may use OpenCV to get the coefficients of a fitting quadratic function). Note that all lane types are to be detectable (could add an indicator that tells us what type the detected lane is). Also note that "marking" is supposed to be semi-transparent highlighting, not opaque coloring.

### Milestones

- [x] _**~Create repository~ (April 16th, 2022)**_
- [x] _**~Create initial README.md~ (April 23rd, 2022)**_
- [ ] Create pipeline and test it against static image samples in assets
  - [ ] Image #1 Pass (straight_lines1.jpg)
  - [ ] Image #2 Pass (straight_lines2.jpg)
  - [ ] Image #3 Pass (test1.jpg)
  - [ ] Image #4 Pass (test2.jpg)
  - [ ] Image #5 Pass (test3.jpg)
  - [ ] Image #6 Pass (test4.jpg)
  - [ ] Image #7 Pass (test5.jpg)
  - [ ] Image #8 Pass (test6.jpg)
- [ ] Test pipeline against video samples in assets
  - [ ] Video #1 Pass (project_video.mp4)
  - [ ] Video #2 Pass (challenge_video.mp4)
  - [ ] Bonus: Video #3 Pass (harder_challenge_video.mp4)
- [ ] Full code clean-up and documentation with debugging
  - [ ] Show the individual image/video processing steps
  - [ ] Show any relevant statistics
- [ ] A simple bash/shell script to call python to run the code with arguments to run in debugging mode...
  - [ ] Example cmd for normal run: `run.sh input_path output_path`
  - [ ] Example cmd for debug run: `run.sh input_path output_path -d`
- [ ] Demo jupyter notebook for demonstrating the features of the pipeline
- [ ] Upload demo result images and videos
- [ ] Review README.md with respect to phase 1
  - [ ] Add/update necessary information to the <a href="#getting-started">Getting Started</a> section (how to install/run the code, etc...)
  - [ ] Add/update necessary information to the <a href="#usage">Usage</a> section (how to actually play with the code)

### Random Ideas:
- Main problem is the first frame and how to detect lanes dynamically from it... Afterwards, it'll be iterative, so no problem then
- Could surround the detected lines by two parallel detected lines and constrain the search within them for the edges per frame, update them per frame too
  - This could keep track of the lanes and identify left from right
  - Also can help when either lane goes poof

<!-- See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues). -->

<p align="right">(<a href="#top">back to top</a>)</p>

## Phase Two Details and Requirements

TBD.

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

<!-- This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
-->
TBD.

<p align="right">(<a href="#top">back to top</a>)</p> 

## Usage

<!-- Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
 -->
 TBD.
 
<p align="right">(<a href="#top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments
* **Course Instructor:** [Prof. Dr. Mahmoud Khalil](https://eng.asu.edu.eg/public/staff/mahmoud.khalil)
* **Course Teaching Assistant:** [Eng. Mahmoud Selim](https://eng.asu.edu.eg/public/staff/mahmoud.selim)
* [Choose an Open Source License](https://choosealicense.com)
* [Markdown Guide](https://www.markdownguide.org)

<p align="right">(<a href="#top">back to top</a>)</p>



###### Distributed under the  GPL-3.0 License. See [`LICENSE`](/LICENSE) for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/vadrif-draco/asufecse483project-simpleperceptionstack.svg?style=for-the-badge
[contributors-url]: https://github.com/vadrif-draco/asufecse483project-simpleperceptionstack/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vadrif-draco/asufecse483project-simpleperceptionstack.svg?style=for-the-badge
[forks-url]: https://github.com/vadrif-draco/asufecse483project-simpleperceptionstack/network/members
[stars-shield]: https://img.shields.io/github/stars/vadrif-draco/asufecse483project-simpleperceptionstack.svg?style=for-the-badge
[stars-url]: https://github.com/vadrif-draco/asufecse483project-simpleperceptionstack/stargazers
[issues-shield]: https://img.shields.io/github/issues/vadrif-draco/asufecse483project-simpleperceptionstack.svg?style=for-the-badge
[issues-url]: https://github.com/vadrif-draco/asufecse483project-simpleperceptionstack/issues

[python-shield]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue
[python-url]: https://www.python.org/
[opencv-shield]: https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white
[opencv-url]: https://opencv.org/
[numpy-shield]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[pandas-shield]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[jupyter-shield]:	https://img.shields.io/badge/Jupyter-e46e32.svg?&style=for-the-badge&logo=Jupyter&logoColor=white
[jupyter-url]: https://jupyter.org/
[colab-shield]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252
[colab-url]: https://colab.research.google.com/

[before-vision]: assets/test_images/test5.jpg
