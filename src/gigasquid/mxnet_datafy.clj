(ns gigasquid.mxnet-datafy
  (:require[clojure.java.io :as io]
           [clojure.string :as string]
           [clojure.datafy :refer [datafy]]
           [clojure.core.protocols :as p]
           [org.apache.clojure-mxnet.module :as m]
           [org.apache.clojure-mxnet.ndarray :as ndarray]
           [mikera.image.core :as img]
           [think.image.pixel :as pixel]))


(def model-dir "model")
(def num-channels 3)
(def h 224)
(def w 224)

(defn process-image [file]
  (let [image (-> (img/load-image file)
                  (img/resize h w))
        pixels (img/get-pixels image)
        rgb-pixels (reduce (fn [result pixel]
                             (let [[rs gs bs] result
                                   [r g b _] (pixel/unpack-pixel pixel)]
                               [(conj rs r) (conj gs g) (conj bs b)]))
                           [[] [] []]
                           pixels)]
    (img/show image)
    (-> rgb-pixels
        (flatten)
        (ndarray/array [1 num-channels h w]))))


(defn predict [file]
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/resnet-152") :epoch 0})
        labels (-> (slurp (str model-dir "/synset.txt"))
                   (string/split #"\n"))
        nd-img (process-image file)
        prob (-> mod
                 (m/bind {:for-training false :data-shapes [{:name "data" :shape [1 num-channels h w]}]})
                 (m/forward {:data [nd-img]})
                 (m/outputs)
                 (ffirst))
        prob-with-labels (mapv (fn [p l] {:prob p :label l})
                               (ndarray/->vec prob)
                               labels)]
    (->> (sort-by :prob prob-with-labels)
         (reverse)
         (take 5))))

(defn predict-images [directory]
  (let [files (->> (file-seq (io/file "images"))
                   (filter #(.isFile %)))]
    (map predict files)))

(defn image-file? [f]
  (and (.isFile f)
       (->> (string/split (.getName f) #"\.")
            (last)
            (string/lower-case)
            #{"jpg" "png"})))


(extend-type java.io.File
  p/Datafiable
  (datafy [f]
    (if (image-file? f)
      {:name (.getName f)
       :classification (predict f)}
      {:name (.getName f)})))



(comment

  (require '[gigasquid.mxnet-datafy])
  (in-ns 'gigasquid.mxnet-datafy)
  (file-seq (io/file "images"))


  ;;;;

  (image-file? (io/file "images"))
  (image-file? (io/file "images/pic1.jpg"))
  (predict (io/file "images"))
  (predict (io/file "images/pic1.jpg"))

  (predict-images "images")

  (->> (file-seq (io/file "images"))
       (map datafy)))
