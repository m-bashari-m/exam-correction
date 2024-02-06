import random

import numpy as np
import cv2
from utils import get_contours
from model import Model


class ResultExtractor:
    def __init__(self, test_blocks, numeric_blocks, model_path="model.keras"):
        self.model = Model(model_path)
        self._test_blocks = test_blocks
        self._numeric_blocks = numeric_blocks
        self._numeric_questions_result = {}
        self._test_questions_result = {}

        # self._numeric_questions_result = self._get_numeric_block_result()
        self._test_questions_result = self._get_test_block_result()

    def _get_numeric_block_result(self):
        blocks_result_dict = {}
        for i, block in enumerate(self._numeric_blocks):
            results = self._numeric_result_extractor(block)
            blocks_result_dict.update({(4 * i) + j + 1: results[j] for j in range(results.shape[0])})

        return blocks_result_dict

    def _get_test_block_result(self):
        blocks_result_dict = dict()
        for i, block in enumerate(self._test_blocks):
            results = self._test_result_extractor(block)

            blocks_result_dict.update({(5 * i) + j + 1: results[i] for j in range(len(results))})

        return blocks_result_dict

    def _numeric_result_extractor(self, block):
        questions = self._slice_numeric_block_to_qeustions(block)
        questions = [cv2.resize(question, dsize=(60, 40)) for question in questions]
        questions = np.stack(questions, axis=0)
        preds = self.model.predict_batch(questions)

        return preds

    def _test_result_extractor(self, block):
        answers = []
        questions = self._slice_test_block_to_questions(block)

        for question in questions:
            # cv2.imwrite(f'images/{random.randint(10, 1000)}.jpg', question)
            answer = self._find_answer_from_test_question(question)
            answers.append(answer)
        # print(answers)
        return answers

    def _slice_test_block_to_questions(self, block):
        height, width = block.shape
        slice_height = height // 5
        sliced_images = []
        for i in range(5):
            start_y = i * slice_height
            end_y = start_y + slice_height
            sliced_images.append(block[start_y:end_y, :])

        return np.array(sliced_images)

    def _slice_numeric_block_to_qeustions(self, block):
        questions, question_boxes = [], []
        height, width = block.shape[:2]

        half_width = width // 2
        half_height = height // 2

        questions.append(block[0:half_height, half_width:width])
        questions.append(block[0:half_height, 0:half_width])
        questions.append(block[half_height:height, half_width:width])
        questions.append(block[half_height:height, 0:half_width])

        for i, question in enumerate(questions):
            contours = get_contours(question, select_top_n=10, with_processing=False)
            for c in contours:

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)

                    cropped_image = question[y:y + h, x:x + w]

                    height, width = cropped_image.shape[:2]
                    shrink_factor = 0.05

                    shrink_x = int(shrink_factor * width)
                    shrink_y = int(shrink_factor * height)

                    cropped_image = cropped_image[shrink_y:-shrink_y, shrink_x:-shrink_x]

                    question_boxes.append(cropped_image)
                    break

        return question_boxes

    def _find_answer_from_test_question(self, question):
        choices, result = [], -1
        h, w = question.shape
        new_w = w // 5
        for i in range(0, 4):
            roi = question[:, new_w * i: (i + 1) * new_w]

            choices.append(np.sum(roi))

            result = choices[::-1].index(min(choices)) + 1

        return result

    def get_numeric_result(self):
        return self._numeric_questions_result

    def get_test_result(self):
        return self._test_questions_result
