import re

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add customRoles state and new modal states
state_addition = '''
  const [customRoles, setCustomRoles] = useState<any[]>([]);
  const [isRoleModalOpen, setIsRoleModalOpen] = useState(false);
  const [roleMode, setRoleMode] = useState<'create' | 'edit'>('create');
  const [roleName, setRoleName] = useState('');
  const [editingOldRoleName, setEditingOldRoleName] = useState('');
'''
code = re.sub(r'(const \[accounts, setAccounts\] = useState<any\[\]>\(\[\]\);)', r'\1\n' + state_addition, code)

# 2. Update loadData to handle roles
load_data_new = '''
  const loadData = () => {
    const storedUsersStr = localStorage.getItem('aitrade-users');
    if (storedUsersStr) {
      setAccounts(JSON.parse(storedUsersStr));
    } else {
      setAccounts([]);
    }
    const storedRolesStr = localStorage.getItem('aitrade-roles');
    if (storedRolesStr) {
      setCustomRoles(JSON.parse(storedRolesStr));
    } else {
      const initialRoles = [
        { name: 'super_admin', permissions: SIDEBAR_OPTIONS.map(o => o.id) },
        { name: 'admin', permissions: SIDEBAR_OPTIONS.map(o => o.id) },
        { name: 'user', permissions: ['dashboard', 'trading', 'portfolio'] }
      ];
      setCustomRoles(initialRoles);
      localStorage.setItem('aitrade-roles', JSON.stringify(initialRoles));
    }
  };
'''
code = re.sub(r'const loadData = \(\) => \{[\s\S]*?\};\n', load_data_new, code)

# 3. Add handleRoleSubmit and handleOpenRoleModal
role_handlers = '''
  const handleOpenRoleModal = (role?: any) => {
    if (role) {
      setRoleMode('edit');
      setEditingOldRoleName(role.name);
      setRoleName(role.name);
      setPermissions(role.permissions || []);
    } else {
      setRoleMode('create');
      setRoleName('');
      setPermissions(SIDEBAR_OPTIONS.map(o => o.id));
    }
    setIsRoleModalOpen(true);
  };

  const handleRoleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!roleName) {
      addNotification({ type: 'error', message: 'Please provide role name' });
      return;
    }
    let updatedRoles = [...customRoles];
    if (roleMode === 'create') {
      if (updatedRoles.find(r => r.name.toLowerCase() === roleName.toLowerCase())) {
        addNotification({ type: 'error', message: 'Role already exists' });
        return;
      }
      updatedRoles.push({ name: roleName, permissions });
      addNotification({ type: 'success', message: `Role ${roleName} created successfully!` });
    } else {
      updatedRoles = updatedRoles.map(r => r.name === editingOldRoleName ? { ...r, name: roleName, permissions } : r);
      addNotification({ type: 'success', message: `Role ${roleName} updated successfully!` });
    }
    setCustomRoles(updatedRoles);
    localStorage.setItem('aitrade-roles', JSON.stringify(updatedRoles));
    setIsRoleModalOpen(false);
  };

  const handleDeleteRole = (nameToDelete: string) => {
    if (nameToDelete === 'super_admin' || nameToDelete === 'admin' || nameToDelete === 'user') {
      addNotification({ type: 'error', message: 'Cannot delete default roles' });
      return;
    }
    if (window.confirm(`Are you sure you want to delete role ${nameToDelete}?`)) {
      const updatedRoles = customRoles.filter(r => r.name !== nameToDelete);
      setCustomRoles(updatedRoles);
      localStorage.setItem('aitrade-roles', JSON.stringify(updatedRoles));
      addNotification({ type: 'info', message: `Role ${nameToDelete} deleted` });
    }
  };
'''
code = code.replace('const handleOpenEditModal = (account: any) => {', role_handlers + '\n  const handleOpenEditModal = (account: any) => {')

# 4. Modify handleOpenCreateModal to use activeTab directly for header button
code = code.replace("const handleOpenCreateModal = (forceRole?: 'admin' | 'user') => {", "const handleOpenCreateModal = (forceRole?: string) => {")

# 5. Fix User Modal Header click
header_btn = '''
                <button
                  onClick={() => activeTab === 'roles' ? handleOpenRoleModal() : handleOpenCreateModal()}
'''
code = re.sub(r'<button\s*onClick=\{\(\) => handleOpenCreateModal\(\)\}', header_btn, code)

# 6. Roles Table in the UI
roles_table = '''
              {/* Roles Management View */}
              {activeTab === 'roles' && (
                <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-sm overflow-hidden mt-4">
                  <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg)]">
                    <h3 className="font-semibold text-[var(--text)] flex items-center gap-2">
                      <ShieldCheck size={16} className="text-purple-400" />
                      {customRoles.length} Role(s) Defined
                    </h3>
                  </div>
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-[var(--bg)] border-b border-[var(--border)]">
                        <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider w-20">S.No</th>
                        <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Role Name</th>
                        <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Access Modules</th>
                        <th className="px-6 py-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider text-right w-32">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border)]">
                      {customRoles.map((roleObj, index) => (
                        <tr key={roleObj.name} className="hover:bg-[var(--bg-hover)] transition-colors">
                          <td className="px-6 py-4 text-[var(--text-muted)] font-medium">{index + 1}</td>
                          <td className="px-6 py-4 text-[var(--text)] font-medium capitalize">{roleObj.name.replace('_', ' ')}</td>
                          <td className="px-6 py-4">
                            <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${
                              (roleObj.permissions || []).length === SIDEBAR_OPTIONS.length
                                ? 'bg-emerald-500/10 text-emerald-400'
                                : 'bg-blue-500/10 text-blue-400'
                            }`}>
                              {(roleObj.permissions || []).length} modules
                            </span>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center justify-end gap-2">
                              <button onClick={() => handleOpenRoleModal(roleObj)} className="p-2 text-[var(--text-muted)] hover:text-blue-400 hover:bg-blue-400/10 rounded-md transition-colors" title="Edit Role">
                                <Edit2 size={16} />
                              </button>
                              {roleObj.name !== 'super_admin' && roleObj.name !== 'admin' && roleObj.name !== 'user' && (
                                <button onClick={() => handleDeleteRole(roleObj.name)} className="p-2 text-[var(--text-muted)] hover:text-rose-400 hover:bg-rose-400/10 rounded-md transition-colors" title="Delete Role">
                                  <Trash2 size={16} />
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
'''
# Replace the old Roles Control Panel and Admins Table
old_roles_view = re.compile(r'\{\/\* Roles Control Panel \*\/\}.*?\{\/\* Users Table \*\/\}', re.DOTALL)
code = old_roles_view.sub(roles_table + '\n\n              {/* Users Table */}', code)

# 7. Add Role Modal HTML at the bottom
role_modal_html = '''
        {/* Modal Overlay for Create/Edit Role */}
        {isRoleModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-[fade-in_0.15s_ease-out]">
            <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-xl shadow-2xl w-full max-w-md overflow-hidden transform scale-100 transition-all">
              <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] bg-[var(--bg)]">
                <h3 className="text-lg font-semibold text-[var(--text)]">
                  {roleMode === 'create' ? 'Create New Role' : 'Edit Role'}
                </h3>
                <button onClick={() => setIsRoleModalOpen(false)} className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-hover)] rounded-md transition-colors">
                  <X size={18} />
                </button>
              </div>
              <form onSubmit={handleRoleSubmit} className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Role Name</label>
                  <input
                    type="text"
                    value={roleName}
                    onChange={(e) => setRoleName(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                    placeholder="e.g. manager, analyst"
                    disabled={roleMode === 'edit' && ['super_admin', 'admin', 'user'].includes(roleName)}
                  />
                </div>
                <div className="pt-2">
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-[var(--text-secondary)]">Sidebar Access</label>
                    <button type="button" onClick={() => setPermissions(permissions.length === SIDEBAR_OPTIONS.length ? [] : SIDEBAR_OPTIONS.map(o => o.id))} className="text-xs text-[var(--primary)] hover:underline">
                      {permissions.length === SIDEBAR_OPTIONS.length ? 'Deselect All' : 'Select All'}
                    </button>
                  </div>
                  <div className="bg-[var(--bg)] border border-[var(--border)] rounded-lg p-3 grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                    {SIDEBAR_OPTIONS.map(option => (
                      <label key={option.id} className="flex items-center gap-2 cursor-pointer group">
                        <input
                          type="checkbox"
                          checked={permissions.includes(option.id)}
                          onChange={(e) => {
                            if (e.target.checked) setPermissions([...permissions, option.id]);
                            else setPermissions(permissions.filter(p => p !== option.id));
                          }}
                          className="w-4 h-4 rounded border-[var(--border)] text-[var(--primary)] focus:ring-[var(--primary)] bg-[var(--bg-elevated)]"
                        />
                        <span className="text-sm text-[var(--text-muted)] group-hover:text-[var(--text)] transition-colors">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <div className="pt-4 flex items-center gap-3">
                  <button type="button" onClick={() => setIsRoleModalOpen(false)} className="flex-1 px-4 py-2 rounded-lg font-medium text-[var(--text)] border border-[var(--border)] hover:bg-[var(--bg-hover)] transition-colors">Cancel</button>
                  <button type="submit" className="flex-1 px-4 py-2 rounded-lg font-medium text-white bg-[var(--primary)] hover:bg-[var(--primary-hover)] transition-colors">
                    {roleMode === 'create' ? 'Create Role' : 'Save Changes'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
'''
code = code.replace('        {/* Modal Overlay for Create/Edit User */}', role_modal_html + '\n        {/* Modal Overlay for Create/Edit User */}')

# 8. Update User Modal dropdown to use customRoles
user_role_dropdown = '''
                  <select
                    value={role}
                    onChange={(e) => {
                      const newRole = e.target.value;
                      setRole(newRole as any);
                      const matchedRole = customRoles.find(r => r.name === newRole);
                      if (matchedRole) setPermissions(matchedRole.permissions);
                    }}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors appearance-none"
                  >
                    {customRoles.map(r => (
                      <option key={r.name} value={r.name}>{r.name.replace('_', ' ')}</option>
                    ))}
                  </select>
'''
code = re.sub(r'<select\s+value=\{role\}[\s\S]*?<\/select>', user_role_dropdown, code)

# 9. Update User table default rendering
# Instead of activeTab check, render all users if activeTab == 'user'
code = re.sub(r'accounts\.filter\(a => \(a\.role \|\| \'user\'\) === activeTab\)', 'accounts', code)
code = re.sub(r'\{activeTab\.replace\(\'_\', \' \'\)\} accounts', 'user accounts', code)
code = re.sub(r'account\.role === \'super_admin\' \? \'Super Admin\' : account\.role === \'admin\' \? \'Admin\' : \'User\'', "account.role?.replace('_', ' ') || 'user'", code)

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'w', encoding='utf-8') as f:
    f.write(code)
