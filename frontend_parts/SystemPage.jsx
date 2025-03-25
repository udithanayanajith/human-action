import Webcam from "react-webcam";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Swal from "sweetalert2";
import { useState, useRef, useEffect } from "react";

function SystemPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [videoURL, setVideoURL] = useState("");
  const [randomCategory, setRandomCategory] = useState("");
  const [difficultyLevel, setDifficultyLevel] = useState("");
  const navigate = useNavigate();
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [cameraStream, setCameraStream] = useState(null);
  const [Playe, setplaye] = useState(false);
  const [radamselect, setradamselect] = useState(true);
  const [displayrecode, setdisplayrecode] = useState(true);
  const [countdown, setCountdown] = useState(0);
  const categories = ["catch", "walk", "stand"];
  const categoryVideos = {
    catch: "/videos/catch.mp4",
    walk: "/videos/walk.mp4",
    stand: "/videos/jump.mp4",
  };

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
  };

  const handleNextTask = () => {
    if (!difficultyLevel || difficultyLevel === "No level") {
      Swal.fire({
        title: "Cannot Proceed",
        text: "The system couldn't determine your performance level. Please try again.",
        icon: "error",
        confirmButtonText: "OK",
      });
      return;
    }
    navigate("/grossmotor/next-task", { state: { difficultyLevel } });
  };

  const handleRandomCategory = () => {
    const randomIndex = Math.floor(Math.random() * categories.length);
    setRandomCategory(categories[randomIndex]);
    setPredictions([]);
    setradamselect(false);
    setVideoBlob(null);
    setplaye(false);
    setdisplayrecode(true);
  };

  const validateResults = (responseData) => {
    if (responseData.detection_ratio < 0.1) {
      return {
        valid: false,
        message: "No action detected in the video. Please try again.",
      };
    }
    if (responseData.score === 0) {
      return {
        valid: false,
        message: "Could not determine performance level. Please try again.",
      };
    }
    return { valid: true };
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
      setDifficultyLevel(level);

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

      if (level == "No level" || level == "No detection") {
        Swal.fire({
          title: `Oops...`,
          html: `<b>Match Percentage:</b> ${percentage}% <br><b>Message:</b> ${message}`,
          icon: "error",
          confirmButtonText: "Try Again",
          confirmButtonColor: "red",
          timerProgressBar: true,
        });
        setVideoBlob(null);
      } else {
        Swal.fire({
          title: `Results Processed!`,
          html: `<b>Match Percentage:</b> ${percentage}% <br>
          <b>Actual action is :</b> ${randomCategory} <br>
          <b>Detected action is :</b> ${predicted_action} <br>
          <b>System suggests next level as:</b> <span style="margin: 10px 0; font-size: 1.8rem; color: ${levelColor}">${levelIcon} ${level}</span>
          <br><b>Message:</b> ${message}`,
          icon: "success",
          confirmButtonText: "OK",
          confirmButtonColor: levelColor,
          timerProgressBar: true,
          customClass: {
            popup: "level-popup",
            confirmButton: "level-confirm-button",
          },
        });
        setVideoBlob(null);
      }
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
          setdisplayrecode(false);
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
      setdisplayrecode(false);
      mediaRecorderRef.current.stop();
    }
  };

  const startrecodevedio = () => {
    setplaye(true);
    setdisplayrecode(true);
    setVideoBlob(null);
    setPredictions([]);
    setUploadStatus("");
  };

  const calculateCategoryCounts = () => {
    return predictions.reduce((acc, item) => {
      acc[item.prediction] = (acc[item.prediction] || 0) + 1;
      return acc;
    }, {});
  };

  const categoryCounts = calculateCategoryCounts();

  return (
    <div
      style={{ padding: "20px" }}
      className="bg-[url(https://cdn.pixabay.com/photo/2022/06/22/11/45/background-7277773_1280.jpg)] bg-cover bg-no-repeat bg-center h-[700px] w-full overflow-y-auto"
    >
      {radamselect && (
        <>
          <h1 className="text-9xl font-bold font-fontstle2 mb-4 text-center mt-36">
            Play The Game
          </h1>
          <div className="flex justify-center">
            <button
              onClick={handleRandomCategory}
              className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 transition-all mb-6 w-[400px] mt-10"
              style={{ marginBottom: "10px" }}
            >
              Generate Random Category
            </button>
          </div>
        </>
      )}
      <div
        className={`${
          randomCategory && "bg-white py-16 px-8 shadow-2xl rounded-xl"
        }`}
      >
        <div className="flex flex-row justify-center gap-8">
          <div>
            {randomCategory && (
              <div className="flex flex-col items-center mt-6">
                <h3 className="text-4xl font-extrabold text-pink-500 mb-4 animate-bounce">
                  Start Learning: {randomCategory}
                </h3>
                <div className="relative">
                  <video
                    src={categoryVideos[randomCategory]}
                    controls
                    className="w-full max-w-3xl rounded-3xl shadow-lg border-4 border-blue-400 hover:border-yellow-400 transition-all duration-500"
                  />
                  <div className="absolute top-0 left-0 w-full h-full animate-pulse"></div>
                </div>
              </div>
            )}
          </div>
          <div>
            {randomCategory && (
              <div className="flex flex-col items-center mt-6 bg-gradient-to-b from-blue-100 to-white p-6 rounded-xl shadow-lg">
                <h2 className="text-4xl font-extrabold text-blue-600 mb-4">
                  Live Camera Preview & Recording 🎥
                </h2>
                <div className="w-full flex flex-col items-center">
                  {Playe && displayrecode ? (
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
                              Starting in ........
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
                      <div>
                        <div className="flex flex-col items-center gap-4 mt-5 p-5 bg-blue-50 rounded-xl shadow-lg">
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
                          <p className="text-blue-600 font-semibold">
                            {uploadStatus}
                          </p>
                        </div>
                        <div className="flex justify-center mt-9">
                          <button
                            onClick={startrecodevedio}
                            className="bg-blue-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-blue-700 transition-all"
                          >
                            Start Recording 🚀
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                </div>
                {videoBlob ? (
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
                ) : (
                  <>
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
                            {Object.entries(categoryCounts).map(
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
                        <div className="flex justify-center mt-6 gap-4">
                          <button
                            onClick={handleRandomCategory}
                            className="bg-yellow-500 text-white px-6 py-3 rounded-lg shadow-md hover:bg-yellow-600 transition-all"
                          >
                            Generate Random Category 🎲
                          </button>
                          <button
                            onClick={handleNextTask}
                            className="px-6 py-3 text-lg font-bold text-white bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full shadow-lg hover:from-orange-500 hover:to-yellow-400 transform hover:scale-105 transition-all duration-300 ease-in-out flex items-center justify-center gap-2"
                            disabled={
                              !difficultyLevel || difficultyLevel === "No level"
                            }
                          >
                            🚀 Go to Next Task 🎉
                          </button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      <br />
    </div>
  );
}

export default SystemPage;
