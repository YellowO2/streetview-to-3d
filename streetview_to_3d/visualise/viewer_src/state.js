// Interaction state is independent of scene geometry and DOM panels.
// Mode changes never clear selection, visibility or edit history.
export class ViewerState {
  mode = 'inspect';
  selected = null;
  selectionKind = 'piece';
  hidden = new Set();
  isolated = false;
  threshold = 0;
  groups = [];
  tool = 'translate';
  select(members, kind = 'piece') {
    this.selectionKind = kind;
    this.selected = members;
    if (members) members.forEach((i) => this.hidden.delete(i));
    else this.isolated = false;
  }
  regroup(groups) {
    const anchor = this.selected?.[0];
    this.groups = groups;
    this.selected =
      anchor == null
        ? null
        : this.selectionKind === 'node'
          ? [anchor]
          : groups.find((m) => m.includes(anchor)) || null;
    if (!this.selected) this.isolated = false;
  }
  visible(i) {
    return !this.hidden.has(i) && (!this.isolated || !this.selected || this.selected.includes(i));
  }
  toggleVisibility(members) {
    const visible = members.some((i) => this.visible(i));
    this.isolated = false;
    members.forEach((i) => (visible ? this.hidden.add(i) : this.hidden.delete(i)));
    if (visible && this.selected?.some((i) => members.includes(i))) this.select(null);
  }
  reset() {
    this.mode = 'inspect';
    this.selected = null;
    this.selectionKind = 'piece';
    this.hidden.clear();
    this.isolated = false;
    this.threshold = 0;
    this.groups = [];
    this.tool = 'translate';
  }
}
