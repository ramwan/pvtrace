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

class CountableFaces():
  def __init__(self, counter_faces, id_to_face):
    self._RED = "#FF0000"
    self._red = "#ff0000"

    self.faces_to_id = {} # {(v1, v2, v3): id}
    self.id_to_face = {} # {id: (v1, v2, v3)}
    self.counter_faces = {} # {id: count}
    
    if id_to_face is not None:
      self.id_to_face = id_to_face

      for (key, v) in id_to_face.items():
        self.faces_to_id[v] = key
    
    if counter_faces is not None and id_to_face is not None:
        for (face, colour) in counter_faces:
            if colour == self._RED or colour == self._red:
              self.counter_faces[ self.faces_to_id[face] ] = 0

  def incrementById(self, fid):
    if fid in self.counter_faces:
      self.counter_faces[fid] += 1

  def totalCounted(self):
    s = 0
    for (key, value) in self.counter_faces:
      s += value
    return s

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
    self.counter_faces = CountableFaces(counter_faces, faces_dict)

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
    #lsc.translate((0, 0, 2)) # TODO temp

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

  def _make_dataframe(self):
    df = pd.DataFrame()

    # rays entering the scene
    for ray, event in self._store["entrance_rays"]:
      rep = asdict(ray)
      rep["kind"] = "entrance"
      rep["event"] = event.name.lower()
      df = df.append(rep, ignore_index=True)

    # rays exiting the scene
    for ray, event in self._store["exit_rays"]:
      rep = asdict(ray)
      rep["kind"] = "exit"
      rep["event"] = event.name.lower()
      df = df.append(rep, ignore_index=True)

    self._df = df
    return df

  def _make_counts(self, df):
    if self._counts is not None:
      return self._counts

    # a bunch of facet related things here which I will placeholder
    # to nothing
    components = self._scene.component_nodes
    lights = self._scene.light_nodes
    all_components = {component.name for component in components}
    all_lights = {light.name for light in lights}

    self._counts = pd.DataFrame({
        "Solar In": pd.Series({}),
        "Solar Out": pd.Series({}),
        "Luminescent Out": pd.Series({}),
        "Luminescent In": pd.Series({})
      }, index=[])

    return self._counts

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
      raise ValueError("Run a simulation before callign this method.")
    return {l["name"] for l in self._user_lights}

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

  def simulate(self, n, progress=None, emit_method="kT"):
    if self._scene is None:
      self._make_scene()
    scene = self._scene

    # `simulate` can be called many times to append more rays
    if self._store is None:
      store = {"entrance_rays": [], "exit_rays": [],
               "emit_one": 0, "emit_two": 0,
               "emit_three": 0, "emit_four_plus": 0}

    vis = self._renderer
    count = 0
    for ray in scene.emit(n):
      self._numsims += 1
      history = photon_tracer.follow(scene, ray, emit_method=emit_method)
      rays, events = zip(*history)
      store["entrance_rays"].append((rays[1], events[1]))

      reemission_count = events.count(Event.EMIT)
      if reemission_count == 1:
        store["emit_one"] += 1
      elif reemission_count == 2:
        store["emit_two"] += 1
      elif reemission_count == 3:
        store["emit_three"] += 1
      elif reemission_count >3:
        store["emit_four_plus"] += 1

      if events[-1] in (Event.ABSORB, Event.KILL):
        # final event is a lost store path information at final event
        store["exit_rays"].append((rays[-1], events[-1]))
      elif events[-1] == Event.EXIT:
        # final event hits the world node. store path information at
        # penultimate location

        # if we've hit a surface, increment a counter if need be
        _, dist, [tid] = self.mesh.nearest.on_surface(np.array([rays[-2].position]))
        if close_to_zero(dist):
            self.counter_faces.incrementById(tid)

        store["exit_rays"].append((rays[-2], events[-2]))

      # Update visualiser
      if vis:
        vis.add_history(history, **self._add_history_kwargs)

      # progress callback
      if progress:
        count += 1
        progress(count)

    self._store = store
    print("Tracing finished.")
    print("Preparing results.")
    df = self._make_dataframe()
    df = self.expand_coords(df, "direction")
    df = self.expand_coords(df, "position")
    self._df = df

  # does this actually make sense in our situation?
  def expand_coords(self, df, column):
    """
    Returns a dataframe with coordinate column expanded into components.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe
    column : str
        The column label

    Returns
    -------
    df : pands.DataFrame
        The dataframe with the column expanded.

    Example
    -------
    Given the dataframe:

      df = pd.DataFrame({'position': [(1, 2, 3)]})

    the function will return a new dataframe:

      edf = expand_coords(df, 'position')
      edf == pd.DataFrame({'position_x': [1], 'position_y': [2], 'position_z': [3]})
    """
    coords = np.stack(df[column].values)
    df["{}_x".format(column)] = coords[:, 0]
    df["{}_y".format(column)] = coords[:, 1]
    df["{}_z".format(column)] = coords[:, 2]
    df.drop(columns=column, inplace=True)
    return df

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

  def counts(self):
    df = self._df
    if df is None:
      df = self._make_dataframe()
      df = self.expand_coords(df, "direction")
      df = self.expand_coords(df, "position")
    counts = self._make_counts(df)
    return counts

  # A bunch of things are invalid, due to undetermined geometry
  # representations for now
  def summary(self):
    counts = self._make_counts(self._df)
    lum_collected = 0
    lum_escaped = 0
    incident = 0
    # abunch of things dealing with facets...

    lost = self.spectrum(source="all", events={"absorb"}, kind="last").shape[0]
    # optical_efficiency = lum_collected / incident
    optical_efficiency = 0
    # waveguide_efficiency = lum_collected / (lum_collected + lum_escaped)
    waveguide_efficiency = 0
    # nonradiative_loss = lost / incident
    nonradiative_loss = 0
    total_reemitted = self._store["emit_one"] + self._store["emit_two"] + \
                      self._store["emit_three"] + \
                      self._store["emit_four_plus"]

    Cg = 0 # TODO: geometric concentration

    s = pd.Series(
      {
        #"Optical Efficiency": optical_efficiency,
        #"Waveguide Efficiency": waveguide_efficiency,
        #"Waveguide Efficiency (Thermodynamic Prediction": "invalid",
        #"Non-radiative Loss (fraction)": nonradiative_loss,
        #"Geometric Concentration": "invalid (for now)",
        "Refractive Index": self.n1,
        #"Cell Surfaces": "invalid (for now)",
        "Components": self.component_names(),
        "Lights": self.light_names(),
        "Re-emitted once": str(self._store["emit_one"]),
        "Re-emitted twice": str(self._store["emit_two"]),
        "Re-emitted three times": str(self._store["emit_three"]),
        "Re-emitted four plus": str(self._store["emit_four_plus"]),
        "Total rays re-emitted": str(total_reemitted) + " / " \
                                 + str(self._numsims),
        "Total counted at faces": str(self.counter_faces.totalCounted())
      })
    return s

  def report(self):
    print()
    print("Simulation Report")
    print("-----------------")
    #print()
    #print("Surface Counts:")
    #print(self.counts())
    print()
    print("Summary:")
    print(self.summary())



if __name__ == "__main__":
    pass