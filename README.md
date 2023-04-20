# RecogniChess

## Context
Chess, a centuries-old strategic board game, remains popular among enthusiasts and masters alike. In long-format competitive play, every move executed by each side must be manually recorded using pen and paper. One reason for this is post-game analysis, where chess players will review their games in order to improve their strategies and skills. Another reason is to resolve conflict about illegal moves or to determine draws by number of repetitions, among others. The need for an automatic record of chess moves on physical boards emerges from the time-consuming nature of manual record keeping for both long-format and short-format games. While this is an area where Deep Learning can be applied, there has not been much progress. This can be attributed to the difficulty of gathering enough labelled data, as there are up to 32 pieces and associated locations to label per image. This project proposes an unsupervised approach to automatic annotation of chessboard photographs. The approach centers around the concept of unsupervised domain adaptation, a technique used to improve the performance of a model on a target domain containing no labelled data by using the knowledge learned by the model from a related source domain with an abundance of labelled data with a slightly different distribution. The source domain data employed to perform the domain adaptation consists of 3D images of chessboards rendered using Blender, as these are simple to generate in large numbers and can be designed to match the distribution of the target domain. The target domain data employed was unlabelled top-view photographs of chess positions. From a broader perspective, the proposed solution consists of 3 components: A pre-processing pipeline which takes a full target domain photograph as input, detects the board, and crops out the individual squares. Then, the individual squares are passed one at a time to a Deep Learning model trained using domain adaptation which can classify the labels of the chess pieces on each square. Finally, the ordered predictions of the model are passed to a post-processing pipeline which generates a FEN string representing the position that can be fed to a chess engine to generate a complete 2D representation of the input.

## Running the code
Please use "pip install -r requirements.txt" to install the dependencies required for the project. Please note that the pre-processing pipeline code requires an older version of numpy, as explained using a comment in the header of the preprocessing.ipynb file.

## License
Copyright (C) 2023 Wassim Jabbour and others

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

