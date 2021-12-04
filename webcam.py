import cv2
import face_detection_cv2 as face_cv2
import face_detection_tf as face_tf
import sys, os, getopt
import time


def read_args(argv):
    try:
        opts, args = getopt.getopt(argv, "l:m:", ["label=", "max="])
    except getopt.GetoptError:
        print('example: python webcam.py --label Jokowi --max 5')
        sys.exit(2)

    is_training, label, max = False, '', 5
    for opt, arg in opts:
        if opt in ("-l", "--label"):
            is_training = True
            label = arg
        elif opt in ("-m", "--max"):
            max = arg

    return is_training, label, int(max)


def main(argv):
    is_training, label, max_training = read_args(argv)
    print(f'{is_training}|{label}|{max_training}')

    cam = cv2.VideoCapture(0)
    isShown = False
    training_count = 0

    while cam.isOpened():
        check, frame = cam.read()
        if check:
            # COLOR_BGR2GRAY |COLOR_BGR2RGB
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img_gray = cv2.imread('/Users/annasblackhat/Pictures/jokowi-1.jpeg')
            faces = face_cv2.detect_face(img_gray)
            print(f'{len(faces)} faces detected...')
            for face in faces:
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                face_crop = img_gray[y:y+h , x:x+w]
                age,_ = face_tf.recognize_age(face_crop)
                label, confidence = face_tf.recognize(face_crop, is_training, label)
                conf = ''
                if confidence > 0.0:
                    conf = ' ({:.2f}%)'.format(confidence)
                image_label = label + conf + ' age:' + age

                # text_size, _ = cv2.getTextSize(image_label, cv2.FONT_HERSHEY_PLAIN, 2, 3)
                # text_w, text_h = text_size
                # cv2.rectangle(img_gray, (x, y-10), (x + text_w, (y-10) + text_h), (0,0,0), -1)
                cv2.putText(
                    frame, image_label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)

                if is_training:
                    training_count = training_count + 1
                    os.makedirs(f'temp/faces/{label}/', exist_ok=True)
                    cv2.imwrite(f'temp/faces/{label}/{label}_{time.time()}.png', face_crop)
                    # break

                # if c % 50 == 0:
                #     print('---- SAVING -------')
                #     cv2.imwrite(f'face_{c}.png', face_crop)
            # face_tf.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # if not isShown and c == 50:
            #     cv2.imshow('Image', img_gray)
            #     isShown = True

            cv2.imshow('video', frame)
            if is_training and training_count >= max_training:
                print('### TRAINING FINISH ###')
                break;

            key = cv2.waitKey(10)
            if key == 27:
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
