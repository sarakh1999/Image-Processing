import cv2
import numpy as np
import math
import random


def select_random_block(refrence_texture, block_size):
    x_texture_refrence = refrence_texture.shape[0]
    y_texture_refrence = refrence_texture.shape[1]
    x_search_boundry = x_texture_refrence - block_size
    y_search_boundry = y_texture_refrence - block_size
    selected_x = random.randint(0, x_search_boundry)
    selected_y = random.randint(0, y_search_boundry)
    x1 = selected_x
    x2 = selected_x + block_size
    y1 = selected_y
    y2 = selected_y + block_size
    selected_area = refrence_texture[x1:x2, y1:y2]
    return selected_area


def select_patch_from_down_of_the_block(block, patch_length):
    selected_patch_boundry = block.shape[0] - patch_length
    selected_patch = block[selected_patch_boundry:, :]
    mask = np.zeros((block.shape[0], block.shape[1], 3), dtype=np.uint8)
    mask[:selected_patch.shape[0], :, :] = 255
    return mask, selected_patch


def select_patch_from_right_of_the_block(block, patch_length):
    selected_patch_boundry = block.shape[0] - patch_length
    selected_patch = block[:, selected_patch_boundry:]
    mask = np.zeros((block.shape[0], block.shape[1], 3), dtype=np.uint8)
    mask[:, :selected_patch.shape[1], :] = 255
    return mask, selected_patch


def assemble_patches(left_patch, up_patch, block_size):
    mask = np.zeros((block_size, block_size, 3), dtype=np.uint8)
    mask[:, :left_patch.shape[1], :] = 255
    mask[:up_patch.shape[0], :, :] = 255
    patch = np.zeros((block_size, block_size, 3), dtype=np.uint8)
    patch[:, :left_patch.shape[1], :] = left_patch
    patch[:up_patch.shape[0], :, :] = up_patch
    return mask, patch


def template_matching(refrence_texture, patch, block_size, mask):
    # har area ee ba function e template matching qabele searche
    # faqat kafie bit haye mask e aan nahiye na 0 bashan ta donbale oona begarde
    x, y, _ = refrence_texture.shape
    search_x = x - block_size
    search_y = y - block_size
    search_area = refrence_texture[:search_x, :search_y]
    # print("search_area.shape:    ", search_area.shape)
    # print(mask.shape)
    # print(patch.shape)
    # with cv2.TM_SQDIFF_NORMED template matching method, the less the better
    matches = cv2.matchTemplate(search_area, patch, cv2.TM_SQDIFF_NORMED, mask=mask)
    # If input image is of size (WxH) and template image is of size (wxh), output image will have a size of (W-w+1, H-h+1)
    # print("matches.shape:    ", matches.shape)
    number_of_wanted_top_most_similarities = 4
    positions = []
    for i in range(number_of_wanted_top_most_similarities):
        # cv2.minMaxLoc(matches):
        # get the minimum value of the matches array
        # get the maximum value of the matches array
        # get the position (x and y) of the minimum the matches array
        # get the position (x and y) of the maximum the matches array
        _, _, min_value_position, _ = cv2.minMaxLoc(matches)
        positions.append(min_value_position)
        y1 = min_value_position[1] - patch.shape[0] // 2
        y2 = min_value_position[1] + patch.shape[0] // 2
        start_height = max(0, y1)
        end_height = min(refrence_texture.shape[0], y2)
        x1 = min_value_position[0] - patch.shape[1] // 2
        x2 = min_value_position[0] + patch.shape[1] // 2
        start_length = max(0, x1)
        end_length = min(refrence_texture.shape[1], x2)
        matches[start_height:end_height, start_length:end_length] = np.max(matches)
    # find the best path in most similar blocks
    min_value_position = positions[random.randint(0, len(positions) - 1)]
    x_1 = min_value_position[1]
    x_2 = min_value_position[1] + block_size
    y_1 = min_value_position[0]
    y_2 = min_value_position[0] + block_size
    matched_segment = refrence_texture[x_1:x_2, y_1:y_2]
    # cv2.rectangle(search_area, (x_1, y_1),(x_1 + block_size, y_2), (0, 255, 0), 2)
    # print("matched_segment.shape:   ", matched_segment.shape)
    return matched_segment


def calculate_differences(left_matrix, right_matrix):
    differences = np.sum((left_matrix - right_matrix) ** 2, axis=2, dtype=np.uint32)
    return differences


def find_min_cut(first, second, patch_length):
    diff = calculate_differences(first, second)
    DP = np.zeros(diff.shape, dtype=np.uint32)
    ancestors = np.zeros(diff.shape, dtype=np.uint32) - 1
    DP[0, :] = diff[0, :]
    for row in range(1, DP.shape[0]):
        prev = row - 1
        for node in range(DP.shape[1]):
            minimum_pixel = node
            minimum_value = math.inf
            for parent in range(node - 1, node + 2):
                if 0 <= parent < DP.shape[1]:
                    if DP[prev, parent] < minimum_value:
                        minimum_value = DP[prev, parent]
                        minimum_pixel = parent
            DP[row, node] = minimum_value + diff[row, node]
            ancestors[row, node] = minimum_pixel

    block_path = np.full((diff.shape[0], diff.shape[0]), 255, dtype=np.uint8)
    arr = np.arange(DP.shape[1])
    minimum_arr_value = np.argmin(DP[-1, :])
    mask = block_path[:, :patch_length]
    for row in reversed(range(DP.shape[0])):
        mask[row][arr < minimum_arr_value] = 0
        minimum_arr_value = ancestors[row][minimum_arr_value]
    patch_path = block_path[:, :patch_length]
    return block_path, patch_path


def fill_first_row(reference_texture, previous_block, patch_length, block_size):
    left_mask, left_patch = select_patch_from_right_of_the_block(previous_block, patch_length)
    b = cv2.cvtColor(left_patch, cv2.COLOR_LAB2BGR)
    cv2.imwrite('left_patch.jpg', b)
    cv2.imwrite('left_mask.jpg', left_mask)
    fit_left_mask = left_mask[:, :patch_length, :]
    cv2.imwrite('fit_left_mask.jpg', fit_left_mask)
    right_matched_block = template_matching(reference_texture, left_patch, block_size, fit_left_mask)
    # print(right_matched_block.shape)
    right_matched_patch = right_matched_block[:, :patch_length, :]
    # print(right_matched_patch.shape)
    block_path, patch_path = find_min_cut(left_patch, right_matched_patch, patch_length)
    cv2.imwrite('block_path-amoodi.jpg', block_path)
    cv2.imwrite('patch_path-amoodi.jpg', patch_path)
    new_block = np.zeros((block_size, block_size, 3), dtype=np.uint8)
    # print(patch_path.shape)
    for i in range(patch_path.shape[0]):
        for j in range(patch_path.shape[1]):
            if patch_path[i][j] == 0:
                new_block[i][j][:] = left_patch[i][j][:]
            elif patch_path[i][j] == 255:
                new_block[i][j][:] = right_matched_patch[i][j][:]
    # print("here")
    # print(new_block.shape)
    # print(right_matched_block.shape)
    new_block[:, patch_length:, :] = right_matched_block[:, patch_length:, :]
    a = cv2.cvtColor(new_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('new_block_ofoqi.jpg', a)
    return new_block


def fill_first_block_of_other_rows(reference_texture, previous_block, patch_length, block_size):
    up_mask, up_patch = select_patch_from_down_of_the_block(previous_block, patch_length)
    c = cv2.cvtColor(up_patch, cv2.COLOR_LAB2BGR)
    cv2.imwrite('up_patch.jpg', c)
    cv2.imwrite('up_mask.jpg', up_mask)
    fit_up_mask = up_mask[:patch_length, :, :]
    cv2.imwrite('fit_up_mask.jpg', fit_up_mask)
    down_matched_block = template_matching(reference_texture, up_patch, block_size, fit_up_mask)
    c = cv2.cvtColor(down_matched_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('down_matched_block.jpg', c)
    down_matched_patch = down_matched_block[:patch_length, :, :]
    block_path, patch_path = find_min_cut(np.transpose(up_patch, (1, 0, 2)),
                                          np.transpose(down_matched_patch, (1, 0, 2)),
                                          patch_length)
    block_path = block_path.T
    patch_path = patch_path.T
    # print(patch_path.shape)
    cv2.imwrite('block_path-ofoqi.jpg', block_path)
    cv2.imwrite('patch_path-ofoqi.jpg', patch_path)

    new_block = np.zeros((block_size, block_size, 3), dtype=np.uint8)
    for i in range(patch_path.shape[0]):
        for j in range(patch_path.shape[1]):
            if patch_path[i][j] == 0:
                new_block[i][j][:] = up_patch[i][j][:]
            elif patch_path[i][j] == 255:
                new_block[i][j][:] = down_matched_patch[i][j][:]
    new_block[patch_length:, :, :] = down_matched_block[patch_length:, :, :]
    a = cv2.cvtColor(new_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('new_block_amoodi.jpg', a)
    return new_block


def fill_other_parts_of_other_rows(reference_texture, previous_upper_block, previous_left_block, patch_length,
                                   block_size):
    left_mask, left_patch = select_patch_from_right_of_the_block(previous_left_block, patch_length)
    a = cv2.cvtColor(previous_upper_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('previous_upper_block.jpg', a)
    a = cv2.cvtColor(previous_left_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('previous_left_block.jpg', a)
    up_mask, up_patch = select_patch_from_down_of_the_block(previous_upper_block, patch_length)
    a = cv2.cvtColor(up_patch, cv2.COLOR_LAB2BGR)
    cv2.imwrite('up_patch.jpg', a)
    a = cv2.cvtColor(left_patch, cv2.COLOR_LAB2BGR)
    cv2.imwrite('left_patch.jpg', a)

    mask, patch = assemble_patches(left_patch, up_patch, block_size)
    cv2.imwrite('mask.jpg', mask)
    a = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)
    cv2.imwrite('patch.jpg', a)
    matched_block = template_matching(reference_texture, patch, block_size, mask)
    a = cv2.cvtColor(matched_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('matched_block.jpg', a)

    right_matched_patch = matched_block[:, :patch_length, :]
    block_path0, patch_path0 = find_min_cut(left_patch, right_matched_patch, patch_length)
    cv2.imwrite('block_path-amoodi.jpg', block_path0)
    cv2.imwrite('patch_path-amoodi.jpg', patch_path0)
    new_block = np.zeros((block_size, block_size, 3), dtype=np.uint8)

    down_matched_patch = matched_block[:patch_length, :, :]
    block_path1, patch_path1 = find_min_cut(np.transpose(up_patch, (1, 0, 2)),
                                            np.transpose(down_matched_patch, (1, 0, 2)),
                                            patch_length)
    block_path1 = block_path1.T
    patch_path1 = patch_path1.T
    cv2.imwrite('block_path-ofoqi.jpg', block_path1)
    cv2.imwrite('patch_path-ofoqi.jpg', patch_path1)
    patch_path = block_path0 & block_path1
    cv2.imwrite('patch_path.jpg', patch_path)

    # patch , matched_block
    for i in range(patch_path.shape[0]):
        for j in range(patch_path.shape[1]):
            if patch_path[i][j] == 0:
                new_block[i][j] = patch[i][j]
            elif patch_path[i][j] == 255:
                new_block[i][j] = matched_block[i][j]

    a = cv2.cvtColor(new_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('new_block.jpg', a)
    a = new_block - matched_block
    a = np.where(a != 0, 255, 0)
    cv2.imwrite('a.jpg', a)
    return new_block


def make_textures(reference_texture, desired_texture_size):
    synthesized_texture = np.zeros((result_img_size, result_img_size, 3), dtype=np.uint8)
    # fill the first block in first row
    first_random_block = select_random_block(reference_texture, block_size)
    a = cv2.cvtColor(first_random_block, cv2.COLOR_LAB2BGR)
    cv2.imwrite('first_random_block.jpg', a)
    synthesized_texture[0:first_random_block.shape[0], 0:first_random_block.shape[1], :] = first_random_block
    ###################################################################################################
    # fill first row
    previous_block = first_random_block
    for i in range(1, number_of_blocks_per_row + 1):
        new_block = fill_first_row(reference_texture, previous_block, patch_length, block_size)
        synthesized_texture[:block_size, i * delta:(i + 1) * delta + patch_length, :] = new_block
        previous_block = new_block
    a = cv2.cvtColor(synthesized_texture, cv2.COLOR_LAB2BGR)
    cv2.imwrite('first_row_filled.jpg', a)
    ###################################################################################################
    # fill the first block of each row
    previous_block = first_random_block
    for i in range(1, number_of_blocks_per_row + 1):
        new_block = fill_first_block_of_other_rows(reference_texture, previous_block, patch_length, block_size)
        synthesized_texture[i * delta:(i + 1) * delta + patch_length, :block_size, :] = new_block
        previous_block = new_block
    a = cv2.cvtColor(synthesized_texture, cv2.COLOR_LAB2BGR)
    cv2.imwrite('first_row_and_column_filled.jpg', a)
    # ###################################################################################################
    # fill other rows
    for i in range(1, number_of_blocks_per_row + 1):
        for j in range(1, number_of_blocks_per_row + 1):
            previous_upper_block = synthesized_texture[(j - 1) * delta:(j) * delta + patch_length,
                                   i * delta:(i + 1) * delta + patch_length, :]
            previous_left_block = synthesized_texture[j * delta:(j + 1) * delta + patch_length,
                                  (i - 1) * delta:(i) * delta + patch_length,
                                  :]
            new_block = fill_other_parts_of_other_rows(reference_texture, previous_upper_block, previous_left_block,
                                                       patch_length,
                                                       block_size)
            synthesized_texture[j * delta:(j + 1) * delta + patch_length, i * delta:(i + 1) * delta + patch_length,
            :] = new_block

    # ###################################################################################################
    # save the result
    a = cv2.cvtColor(synthesized_texture, cv2.COLOR_LAB2BGR)
    cv2.imwrite('synthesized_texture.jpg', a)
    result = a[:desired_texture_size, :desired_texture_size, :]
    # numbers used in this section are just for showing the main texture and the synthesized one in a same picture
    #and they are just offsets set manually
    whole_pic = np.full((desired_texture_size + 2000, desired_texture_size + 2000, 3),255, dtype=np.uint8)
    reference_texture = cv2.cvtColor(reference_texture, cv2.COLOR_LAB2BGR)
    whole_pic[200:result.shape[0] + 200, 200:result.shape[1] + 200, :] = result
    whole_pic[result.shape[0] - 400 :reference_texture.shape[0] + result.shape[0] - 400 , result.shape[1] + 400:reference_texture.shape[1] + result.shape[1] + 400,
    :] = reference_texture
    return whole_pic


# initializations
result_img_size = 2600
desired_texture_size = 2500
block_size = 120
patch_length = block_size // 4
delta = block_size - patch_length
number_of_blocks_per_row = (result_img_size - block_size) // (block_size - patch_length)

# result 1
reference_texture = cv2.imread('texture03.jpg')
reference_texture = cv2.cvtColor(reference_texture, cv2.COLOR_BGR2LAB)
result11 = make_textures(reference_texture, desired_texture_size)
cv2.imwrite("res11.jpg", result11)

#result 2
reference_texture = cv2.imread('texture06.jpg')
reference_texture = cv2.cvtColor(reference_texture, cv2.COLOR_BGR2LAB)
result11 = make_textures(reference_texture,desired_texture_size)
cv2.imwrite("res12.jpg",result11)

# result 3
reference_texture = cv2.imread('texture15.jpg')
reference_texture = cv2.cvtColor(reference_texture, cv2.COLOR_BGR2LAB)
result11 = make_textures(reference_texture,desired_texture_size)
cv2.imwrite("res13.jpg",result11)

#result 4
reference_texture = cv2.imread('texture17.jpg')
reference_texture = cv2.cvtColor(reference_texture, cv2.COLOR_BGR2LAB)
result11 = make_textures(reference_texture,desired_texture_size)
cv2.imwrite("res14.jpg",result11)
