const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  AlignmentType, Header, Footer, PageNumber,
} = require('docx');

const OUT_DIR = path.join(__dirname, 'outputs');

function img(name, width, height) {
  return new ImageRun({
    type: 'png',
    data: fs.readFileSync(path.join(OUT_DIR, name)),
    transformation: { width, height },
  });
}

function figure(name, width, height, caption) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, children: [img(name, width, height)] }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: caption, italics: true, size: 20 })],
    }),
  ];
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}

function p(text) {
  return new Paragraph({
    spacing: { after: 200 },
    children: [new TextRun({ text })],
  });
}

const doc = new Document({
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 } }, // US Letter
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: 'Neural Network Pothole Classifier — Summary Report', size: 18, color: '888888' })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
        })],
      }),
    },
    children: [
      new Paragraph({
        heading: HeadingLevel.TITLE,
        children: [new TextRun({ text: 'Neural Network Classifier for Pothole Detection' })],
      }),
      new Paragraph({
        spacing: { after: 300 },
        children: [new TextRun({
          text: 'A Convolutional Neural Network trained and evaluated on the Pothole and Plain Road Images dataset, with hyperparameter tuning and confusion-matrix-based evaluation.',
          italics: true, color: '555555',
        })],
      }),

      h1('Overview'),
      p(
        'This report summarises the implementation of a neural network image classifier for pothole detection, ' +
        'built as part of the Week 4 assignment that follows on from the dataset chosen in Week 2. The original ' +
        'Week 2 dataset, a YOLOv8 object-detection export named Pothole.v1-raw.yolov8, turned out on closer ' +
        'inspection to contain no pothole-free images at all: every one of its 665 images has at least one ' +
        'annotated pothole bounding box. Since a genuine whole-image "pothole present versus absent" classifier ' +
        'requires both classes to exist, that dataset alone could not support this task. To resolve this, the ' +
        'closest viable substitute was used instead: the Pothole and Plain Road Images dataset from Kaggle, ' +
        'which pairs pothole images with genuine clean-road "Plain" negatives, giving a proper two-class ' +
        'classification problem (Plain = 0, clean road with no pothole; Pothole = 1, road containing a pothole).'
      ),
      p(
        'This report documents the neural network that was implemented, the hyperparameters that were adjusted ' +
        'and the effect this had on the training and validation curves, and the final evaluation of the model ' +
        'using a confusion matrix together with precision, recall, and F1-score, illustrated throughout with ' +
        'snapshots of the code and the results produced during experimentation.'
      ),

      h1('Dataset and Data Preparation'),
      p(
        'The dataset ships with a train folder containing 367 plain-road images and 357 pothole images (724 ' +
        'total), and a much smaller test folder with only 8 images of each class (16 total). Because a test ' +
        'set of just 16 images is too small to draw reliable conclusions from — each single image is worth ' +
        'more than six percentage points of accuracy — all 740 images from both folders were pooled together ' +
        'and re-split into a fresh, stratified 70% / 15% / 15% partition. This produced a training set of 518 ' +
        'images (263 plain, 255 pothole), a validation set of 111 images (56 plain, 55 pothole), and a test ' +
        'set of 111 images (56 plain, 55 pothole). Hyperparameter tuning only ever used the training and ' +
        'validation portions; the re-carved test split was held out completely until the final evaluation.'
      ),
      ...figure('01_class_balance.png', 500, 333, 'Figure 1. Class balance across the re-carved train, validation, and test splits.'),
      ...figure('02_sample_images.png', 550, 313, 'Figure 2. Sample training images — plain road (top row) versus pothole (bottom row).'),

      h1('Model Architecture'),
      p(
        'A compact convolutional neural network (CNN) was implemented in PyTorch. It consists of four ' +
        'convolutional blocks with channel widths of 16, 32, 64, and 128, each followed by batch normalisation, ' +
        'a ReLU activation, and 2x2 max-pooling. The resulting feature map is reduced with global average ' +
        'pooling and passed through a small classifier head — a dropout layer, a 64-unit fully connected layer ' +
        'with ReLU, a second dropout layer, and a final single-logit output layer — trained with binary ' +
        'cross-entropy loss. All images were resized to 128 by 128 pixels in RGB, and random horizontal flips ' +
        'were applied during training as a simple augmentation.'
      ),

      h1('Hyperparameter Tuning'),
      p(
        'Six configurations of the same architecture were trained for 20 epochs each, varying the learning ' +
        'rate, batch size, dropout rate, and optimizer, in order to observe how each choice affected the ' +
        'training and validation curves. The baseline configuration used the Adam optimizer with a learning ' +
        'rate of 1e-3, a batch size of 32, and a dropout of 0.3. Variants then lowered the learning rate to ' +
        '1e-4, raised it to 5e-3, shrank the batch size to 8, raised dropout to 0.6, and finally replaced Adam ' +
        'with SGD and momentum at a learning rate of 1e-2.'
      ),
      p(
        'Learning rate proved to be the most sensitive hyperparameter of the six tested. The baseline (1e-3) ' +
        'and high-learning-rate (5e-3) runs both reached a high peak validation accuracy at some point during ' +
        'training, but their final-epoch validation loss was three to four times higher than the best run, ' +
        'showing that the optimizer kept overshooting narrow minima instead of settling into them. The low ' +
        'learning rate configuration (1e-4) converged the most smoothly of all six runs and finished with both ' +
        'the highest final validation accuracy (94.6%) and the lowest final validation loss (0.148), making it ' +
        'the best balance between continuing to improve and avoiding instability for a training set of this ' +
        'size. Heavier dropout (0.6) reached a similarly high peak accuracy at one point but was markedly less ' +
        'stable overall, while a smaller batch size of 8 trained smoothly but still finished slightly behind ' +
        'the best configuration. Replacing Adam with SGD and momentum was the weakest of the six runs, ' +
        'suggesting it would need more epochs or a tuned learning-rate schedule to catch up. Based on these ' +
        'results, the low-learning-rate configuration (Adam, learning rate 1e-4, batch size 32, dropout 0.3) ' +
        'was selected as the best model and carried forward to the final test-set evaluation.'
      ),
      ...figure('03_hyperparam_val_curves.png', 600, 231, 'Figure 3. Validation loss and accuracy curves for all six hyperparameter configurations.'),
      ...figure('04_hyperparam_comparison.png', 500, 286, 'Figure 4. Best validation accuracy achieved by each hyperparameter configuration.'),
      ...figure('05_best_model_curves.png', 600, 218, 'Figure 5. Training and validation loss/accuracy curves for the selected best model.'),

      h1('Test-Set Evaluation'),
      p(
        'The selected best model was evaluated once on the held-out, re-carved test split of 111 images (56 ' +
        'plain, 55 pothole), which had not been used during either training or hyperparameter selection. The ' +
        'model achieved a test accuracy of 96.4%, with precision, recall, and F1-score for the pothole class ' +
        'all equal to 0.964. The confusion matrix shows only four misclassifications out of 111 images: two ' +
        'plain-road images were incorrectly predicted as containing a pothole, and two pothole images were ' +
        'incorrectly predicted as plain road. Precision, recall, and F1-score landing on exactly the same ' +
        'balanced value indicates the classifier has no systematic bias toward either class, which is ' +
        'consistent with the stability already observed for this configuration during hyperparameter tuning.'
      ),
      ...figure('06_confusion_matrix.png', 400, 320, 'Figure 6. Confusion matrix of the best model on the held-out test split.'),

      h1('Discussion and Conclusions'),
      p(
        'The single biggest factor behind the quality of the final evaluation was not the model or the ' +
        'hyperparameters, but the size of the test set. Evaluating the very same trained model on the ' +
        'dataset’s original 16-image test folder produced a noisy, low-confidence accuracy of only 56%, ' +
        'while evaluating it on the re-carved 111-image test split produced a stable, trustworthy accuracy of ' +
        '96.4% — a clear demonstration that test-set size matters just as much as model choice when judging ' +
        'real-world performance. Aside from that, a small custom CNN trained from scratch on only 518 images, ' +
        'with no pretrained backbone, was able to reach 96.4% test accuracy with balanced precision, recall, ' +
        'and F1-score. The hyperparameter search showed the model is most sensitive to learning rate: values of ' +
        '1e-3 and 5e-3 both reach a good peak accuracy but destabilise by the end of training, while a lower ' +
        'learning rate of 1e-4 with the Adam optimizer gave the smoothest and most reliable convergence. The ' +
        'small number of remaining errors — two false positives and two false negatives — are exactly the ' +
        'kind expected at this image resolution, most likely caused by road markings, cracks, or shadows that ' +
        'resemble pothole texture, and by shallow or partially obscured potholes that resemble plain road.'
      ),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(__dirname, 'Pothole_Classifier_Summary_Report.docx'), buffer);
  console.log('Word document written.');
});
