from pvtrace.material.component import Absorber, Luminophore
from pvtrace.light.light import Light, rectangular_mask
from pvtrace.light.event import Event
from pvtrace.scene.node import Node
from pvtrace.material.material import Material
from pvtrace.material.utils import isotropic, cone, lambertian
from pvtrace.scene.scene import Scene
from pvtrace.geometry.box import Box
from pvtrace.geometry.mesh import Mesh
from pvtrace.geometry.sphere import Sphere
from pvtrace.geometry.utils import EPS_ZERO, close_to_zero
from pvtrace.data import lumogen_f_red_305, fluro_red
from pvtrace.scene.renderer import MeshcatRenderer
from pvtrace.material.surface import Surface, FresnelSurfaceDelegate
from pvtrace.material.distribution import Distribution
from pvtrace.algorithm import photon_tracer
from dataclasses import asdict
import numpy as np
import pandas as pd
import functools
import time
import re

# values: tuple of at least 3 integers representing RGB values
#         tuple can be RGBA but the A will be ignored.
def _isRed(values):
  if values[0] == 255 and values[1] == 0 and values[2] == 0:
    return True
  else:
    return False

def _isBlue(values):
  if values[0] == 0 and values[1] == 0 and values[2] == 255:
    return True
  else:
    return False

# This will probably implement a slightly different interface compared to
# the regular LSC class due to different geometry requirements.
class CustomLSC():
  def __init__(self, mesh, bounding_box,
               wavelength_range=None, n0=1.0, n1=1.5,
               counter_faces=None, faces_dict=None):
    if wavelength_range is None:
      self.wavelength_range = np.arange(400, 800)
    if bounding_box is None:
      raise ValueError("No bounding box dimensions specified for custom LSC.")

    self.mesh = mesh # trimesh mesh object
    self.bounding_box = bounding_box # 3-tuple of values (l, w, d)
    self.n0 = n0
    self.n1 = n1

    self._numsims = 0
    self._scene = None
    self._renderer = None
    self._store = None
    self._df = None
    self._counts = None
    self._user_lights = []
    self._user_components = []

  def _make_default_components(self):
    """
    Default LSC contains Lumogen F Red 305 with concentation
    such that the absorption coefficient at peak is 10 cm-1.
    """
    x = self.wavelength_range
    coefficient = lumogen_f_red_305.absorption(x) * 10.0 # cm-1
    emission = lumogen_f_red_305.emission(x)
    coefficient = np.column_stack((x, coefficient))
    emission = np.column_stack((x, emission))
    lumogen = {
      "cls": Luminophore,
      "name" :"Lumogen F Red 305",
      "coefficient": coefficient,
      "emission": emission,
      "quantum_yield": 1.0,
      "phase_function": None, # will select isotropic
    }
    background = {
      "cls": Absorber,
      "coefficient": 0.1,
      "name": "Background"
    }
    return [lumogen, background]

  def _make_default_lights(self):
    """
    Default light is a spotlight (cone of 20-deg) of single
    wavelength 555nm.
    """
    light = {
      "name": "Light",
      "location": (0.0, 0.0, self.bounding_box[-1] * 0.05),
      "rotation": (np.radians(180), (1, 0, 0)),
      "direction": functools.partial(cone, np.radians(20)),
      "wavelength": None,
      "position": None,
    }
    return [light]

  def _make_scene(self):
    # Creates the scene based on the configuration values
    worlddim = max(self.bounding_box) * 2
    world = Node(
      name="World",
      geometry=Box(
        (worlddim, worlddim, worlddim),
        material=Material(refractive_index=self.n0)
      ))

    # create components (Absorbers, luminophores, scatterers)
    if len(self._user_components) == 0:
      self._user_components = self._make_default_components()
    components = []
    for component_data in self._user_components:
      cls = component_data.pop("cls")
      coefficient = component_data.pop("coefficient")
      component = cls(coefficient, **component_data)
      components.append(component)

    # Create LSC node
    lsc = Node(
      name="LSC-mesh",
      geometry=Mesh(
        trimesh=self.mesh,
        material=Material(
          refractive_index=self.n1,
          components=components,
          surface=Surface())
      ),
      parent=world)

    if len(self._user_lights) == 0:
      self._user_lights = self._make_default_lights()

    for light_data in self._user_lights:
      name = light_data["name"]
      light = Light(
        name=name,
        direction=light_data["direction"],
        wavelength=light_data["wavelength"],
        position=light_data["position"])
      light_node = Node(name=name, light=light, parent=world)
      light_node.location = light_data["location"]
      if light_data["rotation"]:
        light_node.rotate(*light_data["rotation"])

    self._scene = Scene(world)

  # get the names of the components,
  # throw an error if they haven't been created yet
  def component_names(self):
    if self._scene is None:
      raise ValueError("Run a simulation before calling this method.")
    return {c["name"] for c in self._user_components}

  # get the names of the lights
  # throw an error if they haven't been created yet
  def light_names(self):
    if self._scene is None:
      raise ValueError("Run a simulation before calling this method.")
    return {l["name"] for l in self._user_lights}

  def light_names_ordered(self):
    if self._scene is None:
      raise ValueError("Run a simulation before calling this method.")
    return [l["name"] for l in self._user_lights]

  def add_luminophore(
    self, name, coefficient, emission, quantum_yield, phase_function=None
  ):
    self._user_components.append({
      "cls": Luminophore,
      "name": name,
      "coefficient": coefficient,
      "emission": emission,
      "quantum_yield": quantum_yield,
      "phase_function": phase_function,
    })

  def add_absorber(self, name, coefficient):
    self._user_components.append({
      "cls": Absorber,
      "name": name,
      "coefficient": coefficient,
    })

  def add_scatterer(self, name, coefficient, phase_function=None):
    self._user_components.append({
      "cls": Scatterer,
      "name": name,
      "coefficient": coefficient,
      "phase_function": phase_function,
    })

  def add_light(
    self,
    name,
    location,
    rotation=None,
    direction=None,
    wavelength=None,
    position=None
  ):
    self._user_lights.append({
      "name": name,
      "location": location,
      "rotation": rotation,
      "direction": direction,
      "wavelength": wavelength,
      "position": position,
    })

  def show(
    self,
    wireframe=True,
    baubles=True,
    bauble_radius=None, # draws spheres at exit locations
    world_segment="short",
    short_length=None, # the length of final path world_segment when world_segment=short
    open_browser=False,
    max_history=50
  ):
    if bauble_radius is None:
      bauble_radius = np.min(self.bounding_box) * 0.01

    if short_length is None:
      short_length = np.min(self.bounding_box) * 0.2

    self._add_history_kwargs = {
      "bauble_radius": bauble_radius,
      "baubles": baubles,
      "world_segment": world_segment,
      "short_length": short_length,
    }

    if self._scene is None:
      self._make_scene()

    self._renderer = MeshcatRenderer(
      open_browser=open_browser,
      transparency=False,
      opacity=0.5,
      wireframe=wireframe,
      max_histories=max_history,
    )

    self._renderer.render(self._scene)
    time.sleep(1.0)
    return self._renderer

  def simulate(self, n, progress=None, emit_method="kT", update_vis=True):
    numlights = len(self._user_lights)
    vis = self._renderer
    count = 0

    if self._scene is None:
      self._make_scene()
    scene = self._scene

    # `simulate` can be called many times to append more rays
    if self._store is None:
      store = []
      for i in range( numlights ):
        store.append({"entrance_rays": [], "exit_rays": [],
                      "emit_one": 0, "emit_two": 0,
                      "emit_three": 0, "emit_four_plus": 0,
                      "PVcell_count": {}
                    })

    # we're going to emit sequentially because my brain doesn't have time
    # to figure out how to parallelise this and make it fit with the
    # framework which has been written for sequential processing.
    #
    # plus it'd probably take me longer to parallelise it than to just wait.
    i = 0
    for ray in scene.emit(n):
      cur_light = i % numlights
      self._numsims += 1

      if self._numsims % 500 == 0:
        print("Simulated: " + str(self._numsims) + "/" + str(n))

      history = photon_tracer.follow(\
                  scene, ray, maxsteps=10000, emit_method=emit_method)
      rays, events = zip(*history)
      # do we really need to track entrances?
      # store[cur_light]["entrance_rays"].append((rays[1], events[1]))

      reemission_count = events.count(Event.EMIT)
      if reemission_count == 1:
        store[cur_light]["emit_one"] += 1
      elif reemission_count == 2:
        store[cur_light]["emit_two"] += 1
      elif reemission_count == 3:
        store[cur_light]["emit_three"] += 1
      elif reemission_count >3:
        store[cur_light]["emit_four_plus"] += 1

      if events[-1] in (Event.ABSORB, Event.KILL):
        pass
        # final event is a lost store path information at final event
        #store[cur_light]["exit_rays"].append((rays[-1], events[-1]))
      elif events[-1] == Event.EXIT:
        # final event hits the world node. store path information at
        # penultimate location

        # if we've hit a surface, increment a counter if need be.
        # NOTE: this does not include rays that are totally internally
        # reflected from the solar cell boundary.
        _, dist, [tid] = self.mesh.nearest.on_surface(np.array([rays[-2].position]))
        if _isRed(self.mesh.visual.face_colors[tid]):
          if tid in store[cur_light]["PVcell_count"]:
            store[cur_light]["PVcell_count"][tid] += 1
          else:
            store[cur_light]["PVcell_count"][tid] = 1

        # do we really need to track entrances and exits?
        # store[cur_light]["exit_rays"].append((rays[-2], events[-2]))

      # Update visualiser - but we won't want to do this for large
      # numbers of rays
      if vis and update_vis:
        vis.add_history(history, **self._add_history_kwargs)

      # progress callback
      if progress:
        count += 1
        progress(count)

      i += 1
    # end for loop

    self._store = store
    print("Tracing finished.")
    print("Preparing results.")

  # facets isn't well defined for us here...
  def spectrum(self, kind="last", source="all", events=None):
    if self._df is None:
      raise ValueError("Run a simulation before calling this method.")

    df = self._df

    if kind is not None:
      if not kind in {"first", "last"}:
        raise ValueError("Direction must be either `'first'` or `'last'.`")

    if kind is None:
      want_kind = True # Opt-out
    else:
      if kind == "first":
        want_kind = df["kind"] == "entrance"
      else:
        want_kind = df["kind"] == "exit"

    all_sources = self.component_names() | self.light_names()
    if source == "all":
      want_sources = all_sources
    else:
      if isinstance(source, str):
        source = {source}
      if not set(source).issubset(all_sources):
        unknown_source_set = set(source).difference(all_sources)
        raise ValueError("Unknown source requested.", unknown_source_set)

    if source == "all":
      want_source = df["source"].isin(all_sources)
    else:
      want_source = df["source"].isin(set(source))

    if events is None:
      want_events = True # Don't filter by events
    else:
      all_events = {e.name.lower() for e in Event}
      if isinstance(events, (list, tuple, set)):
        events = set(events)
        if events.issubset(all_events):
          want_events = df["event"].isin(events)
        else:
          raise ValueError(
            "Contained some unknown events",
            {"got": events, "expected": all_events})
      else:
        raise ValueError("Events must be a set of event strings",
          {"allowed": all_events})

    return df.loc[want_kind & want_source & want_events]["wavelength"]

  # A bunch of things are invalid, due to undetermined geometry
  # representations for now
  def summary(self):
    lum_collected = 0
    lum_escaped = 0
    incident = 0

    # lost = self.spectrum(source="all", events={"absorb"}, kind="last").shape[0]
    # optical_efficiency = lum_collected / incident
    optical_efficiency = 0
    # waveguide_efficiency = lum_collected / (lum_collected + lum_escaped)
    waveguide_efficiency = 0
    # nonradiative_loss = lost / incident
    nonradiative_loss = 0
    # total_reemitted = self._store["emit_one"] + self._store["emit_two"] + \
    #                   self._store["emit_three"] + \
    #                   self._store["emit_four_plus"]
    total_relevant = 0

    Cg = 0 # TODO: geometric concentration

    series = {}
    names = self.light_names_ordered()

    print("{:28}, {}, {}, {}, {}, {}".format(\
          'Location', 'Re-emit 1x', 'Re-emit 2x',\
          'Re-emit 3x', 'Re-emit 4x+', 'PV-cell count'))
    format_s = "({:08.4f} {:08.4f} {:08.4f}), {:10d}, {:10d}, {:10d}, " +\
               "{:11d}, {:13d}"

    for i in range( len(self._store) ):
      t = 0
      for key in self._store[i]["PVcell_count"]:
        t += self._store[i]["PVcell_count"][key]

      total_relevant += t
      print(format_s.format(\
            *(float(re.sub(r'[^0-9\.]', '', n)) for n in names[i].split(', ')),
            self._store[i]["emit_one"],
            self._store[i]["emit_two"], self._store[i]["emit_three"],
            self._store[i]["emit_four_plus"], t
          ))

    print()
    print("Total counted at PV cell faces: " + str(total_relevant))
    print("Total rays emited: " + str(self._numsims))


if __name__ == "__main__":
    pass
