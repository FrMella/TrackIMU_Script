from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy


index_boundaries = 100, 1000
lang = "ES"  # "ES" espanol  o "EN" English
sample_rate = 100  # 100 Hz

data = numpy.genfromtxt("dataC.csv", delimiter=",", skip_header=index_boundaries[0], max_rows=index_boundaries[1])

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]

figure, axes = pyplot.subplots(nrows=6, sharex=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})

if lang == "ES":
    figure.suptitle("Datos IMU, Angulos Euler, and AHRS estados insternos")
else:
    figure.suptitle("Data From IMU(Euler Angles & AHRS internal states)")

axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="Gyro X")
axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Gyro Y")
axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Gyro Z")
if lang == "ES":
    axes[0].set_ylabel("Grados/s")
else:
    axes[0].set_ylabel("Degrees/s")

axes[0].grid()
axes[0].legend()

axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="Axl X")
axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Axl Y")
axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Axl Z")
axes[1].set_ylabel("g")
axes[1].grid()
axes[1].legend()

offset = imufusion.Offset(sample_rate)  # instanciar ahrs
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(0.5,  # Ganancia
                                   10,  # empuje acceleracion
                                   0,  # empuje magnetico
                                   5 * sample_rate)  # empuje tiempo fuera en segundos

delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

# index -> len(timestamp)
print("elementos timestamp rango: {} a {}".format(index_boundaries[0], len(timestamp)))
for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])
    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])
    euler[index] = ahrs.quaternion.to_euler()
    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_rejection_timer])

    acceleration[index] = 9.81 * ahrs.earth_acceleration  # conversion de g a m/s/s

axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")

if lang == "ES":
    axes[2].set_ylabel("Grados")
else:
    axes[2].set_ylabel("Degrees")
axes[2].grid()
axes[2].legend()

if lang == "ES":
    axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Aceleracion error")
    axes[3].set_ylabel("Grados")
else:
    axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
    axes[3].set_ylabel("Degrees")

axes[3].grid()
axes[3].legend()

if lang == "ES":
    axes[4].plot(timestamp, internal_states[:, 1], "tab:cyan", label="Accelerometro ignorado")
else:
    axes[4].plot(timestamp, internal_states[:, 1], "tab:cyan", label="Accelerometer ignored")
pyplot.sca(axes[4])
pyplot.yticks([0, 1], ["False", "True"])
axes[4].grid()
axes[4].legend()

if lang == "ES":
    axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="reloj de rechazo del accelerometro")
else:
    axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration rejection timer")
axes[5].set_xlabel("Seconds")
axes[5].grid()
axes[5].legend()
_, axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})

axes[0].plot(timestamp, acceleration[:, 0], "tab:red", label="X")
axes[0].plot(timestamp, acceleration[:, 1], "tab:green", label="Y")
axes[0].plot(timestamp, acceleration[:, 2], "tab:blue", label="Z")

if lang == "ES":
    axes[0].set_title("Aceleracion")
else:
    axes[0].set_title("Acceleration")

axes[0].set_ylabel("m/s/s")
axes[0].grid()
axes[0].legend()

is_moving = numpy.empty(len(timestamp))

for index in range(len(timestamp)):
    is_moving[index] = numpy.sqrt(acceleration[index].dot(acceleration[index])) > 3

margin = int(0.1 * sample_rate)  # 100ms

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])

if lang == "ES":
    axes[1].plot(timestamp, is_moving, "tab:cyan", label="moviendose")
else:
    axes[1].plot(timestamp, is_moving, "tab:cyan", label="Moving")

pyplot.sca(axes[1])
pyplot.yticks([0, 1], ["False", "True"])
axes[1].grid()
axes[1].legend()

velocity = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    if is_moving[index]:
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

is_moving_diff = numpy.diff(is_moving, append=is_moving[-1])


@dataclass
class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1


is_moving_periods = []
is_moving_period = IsMovingPeriod()

for index in range(len(timestamp)):
    if is_moving_period.start_index == -1:
        if is_moving_diff[index] == 1:
            is_moving_period.start_index = index

    elif is_moving_period.stop_index == -1:
        if is_moving_diff[index] == -1:
            is_moving_period.stop_index = index
            is_moving_periods.append(is_moving_period)
            is_moving_period = IsMovingPeriod()

velocity_drift = numpy.zeros((len(timestamp), 3))

for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index

    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]

    t_new = timestamp[start_index:(stop_index + 1)]

    velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
    velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
    velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

velocity = velocity - velocity_drift

axes[2].plot(timestamp, velocity[:, 0], "tab:red", label="X")
axes[2].plot(timestamp, velocity[:, 1], "tab:green", label="Y")
axes[2].plot(timestamp, velocity[:, 2], "tab:blue", label="Z")
if lang == "ES":
    axes[2].set_title("Velocidad")
else:
    axes[2].set_title("Velocity")

axes[2].set_ylabel("m/s")
axes[2].grid()
axes[2].legend()

position = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

axes[3].plot(timestamp, position[:, 0], "tab:red", label="X")
axes[3].plot(timestamp, position[:, 1], "tab:green", label="Y")
axes[3].plot(timestamp, position[:, 2], "tab:blue", label="Z")

if lang == "ES":
    axes[3].set_title("Posicion")
    axes[3].set_xlabel("Segundos")
else:
    axes[3].set_title("Position")
    axes[3].set_xlabel("Seconds")

axes[3].set_ylabel("m")
axes[3].grid()
axes[3].legend()

print("Error: " + "{:.3f}".format(numpy.sqrt(position[-1].dot(position[-1]))) + " m")

if True:
    figure = pyplot.figure(figsize=(10, 10))

    axes = pyplot.axes(projection="3d")
    axes.set_xlabel("m")
    axes.set_ylabel("m")
    axes.set_zlabel("m")

    x = []
    y = []
    z = []

    scatter = axes.scatter(x, y, z)

    fps = 30
    samples_per_frame = int(sample_rate / fps)


    def update(frame):
        index = frame * samples_per_frame

        axes.set_title("{:.3f}".format(timestamp[index]) + " s")

        x.append(position[index, 0])
        y.append(position[index, 1])
        z.append(position[index, 2])

        scatter._offsets3d = (x, y, z)

        if (min(x) != max(x)) and (min(y) != max(y)) and (min(z) != max(z)):
            axes.set_xlim3d(min(x), max(x))
            axes.set_ylim3d(min(y), max(y))
            axes.set_zlim3d(min(z), max(z))

            axes.set_box_aspect((numpy.ptp(x), numpy.ptp(y), numpy.ptp(z)))

        return scatter


    anim = animation.FuncAnimation(figure, update,
                                   frames=int(len(timestamp) / samples_per_frame),
                                   interval=1000 / fps,
                                   repeat=False)

    anim.save("figura.gif", writer=animation.PillowWriter(fps))

pyplot.show()
