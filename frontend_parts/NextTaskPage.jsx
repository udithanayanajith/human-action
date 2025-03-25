import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { useLocation, useNavigate } from "react-router-dom";
import Swal from "sweetalert2";

function NextTaskPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { difficultyLevel } = location.state || {};
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [randomCategory, setRandomCategory] = useState("");
  const [recording, setRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [cameraStream, setCameraStream] = useState(null);
  const [countdown, setCountdown] = useState(0);
  const [displayRecord, setDisplayRecord] = useState(false);

  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const categoriesMap = {
    Easy: ["catch", "walk", "stand"],
    Medium: ["run", "throw"],
    Hard: ["dribble", "handstand", "kick-ball"],
  };

  const categoryVideos = {
    catch: "/videos/catch.mp4",
    walk: "/videos/walk.mp4",
    stand: "/videos/jump.mp4",
    run: "/videos/run.mp4",
    throw: "/videos/throw.mp4",
    dribble: "/videos/dribble.mp4",
    handstand: "/videos/handstand.mp4",
    kick_ball: "/videos/kick_ball.mp4",
    somersault: "/videos/somersault.mp4",
  };

  const categories = categoriesMap[difficultyLevel] || [];

  useEffect(() => {
    if (
      webcamRef.current &&
      webcamRef.current.video &&
      webcamRef.current.video.srcObject
    ) {
      setCameraStream(webcamRef.current.video.srcObject);
    }
  }, [webcamRef]);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setVideoBlob(null);
  };

  const handleRandomCategory = () => {
    if (categories.length === 0) {
      Swal.fire(
        "Error!",
        "No categories available for this difficulty level.",
        "error"
      );
      return;
    }
    const randomIndex = Math.floor(Math.random() * categories.length);
    setRandomCategory(categories[randomIndex]);
    setPredictions([]);
    setVideoBlob(null);
    setDisplayRecord(true);
  };

  const handleFileUpload = async (file) => {
    if (!file) {
      Swal.fire(
        "Warning!",
        "Please select or record a video first!",
        "warning"
      );
      return;
    }

    if (!randomCategory) {
      Swal.fire(
        "Warning!",
        "Please generate a random category first!",
        "warning"
      );
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("actual_class", randomCategory);

    try {
      setUploadStatus("Uploading and processing...");
      const response = await axios.post(
        "http://127.0.0.1:5000/action/predict",
        formData
      );

      const { predictions: receivedPredictions } = response.data;
      setPredictions(receivedPredictions);
      setUploadStatus("Processing complete!");
      let message = response.data.message;
      let percentage = response.data.percentage;
      let level = response.data.level;
      let predicted_action = response.data.predicted_action;

      // Determine color based on level
      let levelColor;
      let levelIcon;
      if (level === "Easy") {
        levelColor = "#28a745";
        levelIcon = "👍";
      } else if (level === "Medium") {
        levelColor = "#ffc107";
        levelIcon = "✊";
      } else if (level === "Hard") {
        levelColor = "#dc3545";
        levelIcon = "💪";
      }

      if (level === "No level" || level === "No detection") {
        Swal.fire({
          title: `Oops...`,
          html: `<b>Match Percentage:</b> ${percentage}% <br><b>Message:</b> ${message}`,
          icon: "error",
          showDenyButton: true,
          denyButtonText: "Try Again",
          confirmButtonText: "OK",
          confirmButtonColor: "#3085d6",
          timerProgressBar: true,
        });
      } else {
        Swal.fire({
          title: `Results Processed!`,
          html: `<b>Match Percentage:</b> ${percentage}%   <br>
          <b>Actual action is :</b> ${randomCategory} <br>
          <b>Detected action is :</b> ${predicted_action}
          <br><b>Message:</b> ${message} <br>`,
          icon: "success",
          confirmButtonText: "OK",
          confirmButtonColor: levelColor,
          timerProgressBar: true,
          customClass: {
            popup: "level-popup",
            confirmButton: "level-confirm-button",
          },
        });
      }
      setVideoBlob(null);
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadStatus("Failed to process video.");
      Swal.fire({
        title: "Error!",
        text: "Failed to process video. Please try again.",
        icon: "error",
        confirmButtonText: "OK",
      });
    }
  };

  const calculateCategoryCounts = (predictionsList) => {
    return predictionsList.reduce((acc, item) => {
      acc[item.prediction] = (acc[item.prediction] || 0) + 1;
      return acc;
    }, {});
  };

  const startRecording = async () => {
    if (!cameraStream) {
      Swal.fire(
        "Error!",
        "Camera is not available! Please check your webcam settings.",
        "error"
      );
      return;
    }

    setCountdown(5);
    setUploadStatus(`Recording starts in 5 seconds...`);

    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        const newCount = prev - 1;
        if (newCount > 0) {
          setUploadStatus(`Recording starts in ${newCount} seconds...`);
          return newCount;
        } else {
          clearInterval(countdownInterval);
          startActualRecording();
          return 0;
        }
      });
    }, 1000);
  };

  const startActualRecording = () => {
    setRecording(true);
    setVideoBlob(null);
    setUploadStatus("Recording...");

    try {
      mediaRecorderRef.current = new MediaRecorder(cameraStream, {
        mimeType: "video/webm",
      });
      let recordedChunks = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        setVideoBlob(blob);
        setSelectedFile(blob);
        setUploadStatus("Recording complete! Ready to send.");
      };

      mediaRecorderRef.current.start(80);

      setTimeout(() => {
        if (
          mediaRecorderRef.current &&
          mediaRecorderRef.current.state !== "inactive"
        ) {
          mediaRecorderRef.current.stop();
          setRecording(false);
          setDisplayRecord(false);
        }
      }, 8000);
    } catch (error) {
      console.error("Failed to start recording:", error);
      Swal.fire("Error!", "Failed to start recording. Try again.", "error");
      setUploadStatus("");
      setRecording(false);
    }
  };

  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    ) {
      setRecording(false);
      setDisplayRecord(false);
      mediaRecorderRef.current.stop();
    }
  };

  const startRecordingVideo = () => {
    setDisplayRecord(true);
    setVideoBlob(null);
    setPredictions([]);
    setUploadStatus("");
  };

  return (
    <div
      style={{ padding: "20px" }}
      className="bg-[url(https://cdn.pixabay.com/photo/2022/06/22/11/45/background-7277773_1280.jpg)] bg-cover bg-no-repeat bg-center min-h-screen w-full overflow-y-auto"
    >
      <div className="bg-white py-16 px-8 shadow-2xl rounded-xl">
        <div className="flex flex-row justify-center gap-8">
          <div>
            <h1 className="text-4xl font-bold text-blue-700 mb-8 text-center">
              🎯 Next Task - {difficultyLevel} Level
            </h1>

            <button
              onClick={handleRandomCategory}
              className="mt-4 px-6 py-3 bg-yellow-400 text-white font-bold rounded-full shadow-md hover:bg-yellow-500 transition-all duration-300"
            >
              🎲 Generate Random Category
            </button>

            {randomCategory && categoryVideos[randomCategory] && (
              <div className="mt-5">
                <h3 className="text-xl font-semibold text-blue-600">
                  📺 Introduction Video: {randomCategory}
                </h3>
                <video
                  src={categoryVideos[randomCategory]}
                  controls
                  className="w-full max-w-lg rounded-lg shadow-md mt-3"
                />
              </div>
            )}
          </div>
          <div className="flex flex-col items-center gap-4 mt-2 p-5 bg-blue-50 rounded-xl shadow-lg">
            <label className="cursor-pointer bg-yellow-400 text-white font-bold px-5 py-3 rounded-full shadow-md hover:bg-yellow-500 transition-all duration-300 ease-in-out">
              📂 Select Video
              <input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
            <button
              onClick={() => handleFileUpload(selectedFile)}
              className="bg-green-500 text-white font-bold px-6 py-3 rounded-full shadow-lg hover:bg-green-600 transform hover:scale-105 transition-all duration-300 flex items-center gap-2"
            >
              🚀 Upload & Process Video 🎬
            </button>
            <p className="text-blue-600 font-semibold">{uploadStatus}</p>
          </div>

          <div className="flex flex-col items-center bg-gradient-to-b from-blue-100 to-white p-6 rounded-xl shadow-lg">
            <h2 className="text-4xl font-extrabold text-blue-600 mb-4">
              Live Camera Preview & Recording 🎥
            </h2>

            <div className="w-full flex flex-col items-center">
              {displayRecord ? (
                <>
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    onUserMedia={(stream) => setCameraStream(stream)}
                    mirrored={true}
                    className="w-full max-w-md rounded-xl border-4 border-blue-400 shadow-md"
                  />
                  <div className="mt-4">
                    {recording ? (
                      <div>
                        <button
                          onClick={stopRecording}
                          className="bg-red-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-red-700 transition-all"
                        >
                          Stop Recording ⏹️
                        </button>
                        <p className="text-blue-600 font-semibold mt-2">
                          {uploadStatus}
                        </p>
                      </div>
                    ) : countdown > 0 ? (
                      <div>
                        <button
                          disabled
                          className="bg-gray-500 text-white px-6 py-3 rounded-lg shadow-md cursor-not-allowed"
                        >
                          Starting in {countdown}...
                        </button>
                        <p className="text-blue-600 font-semibold mt-2">
                          {uploadStatus}
                        </p>
                      </div>
                    ) : (
                      <button
                        onClick={startRecording}
                        className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-green-700 transition-all"
                      >
                        Start Recording ⏺️
                      </button>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex justify-center mt-9">
                    <button
                      onClick={startRecordingVideo}
                      className="bg-blue-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-blue-700 transition-all"
                    >
                      Start Recording 🚀
                    </button>
                  </div>
                </>
              )}
            </div>

            {videoBlob && (
              <div className="mt-6 w-full max-w-md">
                <h3 className="text-2xl font-semibold text-blue-600">
                  Recorded Video 🎬
                </h3>
                <video
                  src={URL.createObjectURL(videoBlob)}
                  controls
                  className="w-full rounded-xl shadow-md mt-3"
                />
                <button
                  onClick={() => handleFileUpload(videoBlob)}
                  className="mt-4 bg-purple-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-purple-700 transition-all"
                >
                  Upload Recorded Video ⏫
                </button>
              </div>
            )}

            {predictions.length > 0 && (
              <div className="mt-6 w-full">
                <h2 className="text-2xl font-bold text-blue-700 mb-4">
                  Predictions Summary 📊
                </h2>
                <table className="w-full border-collapse shadow-md bg-white rounded-lg overflow-hidden">
                  <thead className="bg-blue-200 text-blue-900">
                    <tr>
                      <th className="p-3">Category</th>
                      <th className="p-3">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(calculateCategoryCounts(predictions)).map(
                      ([category, count], index) => (
                        <tr
                          key={index}
                          className={`${
                            index % 2 === 0 ? "bg-gray-100" : "bg-white"
                          } text-center border-b border-gray-200`}
                        >
                          <td className="p-3">{category}</td>
                          <td className="p-3">{count}</td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default NextTaskPage;
