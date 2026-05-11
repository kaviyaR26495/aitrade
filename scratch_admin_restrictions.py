import re

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add currentUserRole at the top scope
# Let's find `const navigate = useNavigate();` and add it before or after.
if "const currentUserRole = localStorage.getItem('aitrade-current-role');" not in code:
    code = code.replace("const navigate = useNavigate();", "const currentUserRole = localStorage.getItem('aitrade-current-role');\n  const navigate = useNavigate();")

# 2. Filter accounts table
# Find `accounts.length === 0 ? (`
# Wait, let's just replace the map logic.
old_table_map = """                    {accounts.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-6 py-8 text-center text-[var(--text-muted)]">
                          No user accounts created yet. Click "Create New" to add one.
                        </td>
                      </tr>
                    ) : (
                      accounts.map((account, index) => ("""

new_table_map = """                    {accounts.filter(a => currentUserRole === 'admin' ? (a.role !== 'super_admin' && a.role !== 'admin') : true).length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-6 py-8 text-center text-[var(--text-muted)]">
                          No user accounts created yet. Click "Create New" to add one.
                        </td>
                      </tr>
                    ) : (
                      accounts.filter(a => currentUserRole === 'admin' ? (a.role !== 'super_admin' && a.role !== 'admin') : true).map((account, index) => ("""
code = code.replace(old_table_map, new_table_map)


# 3. Filter Role dropdown in BOTH Create User Modal AND Create Role Modal?
# Wait, Create Role modal doesn't have a Role Dropdown, it has a text input for Role Name.
# But for Create User Modal:
old_dropdown = """                    {customRoles.map(r => (
                      <option key={r.name} value={r.name}>{r.name.replace('_', ' ')}</option>
                    ))}"""

new_dropdown = """                    {customRoles
                      .filter(r => currentUserRole === 'admin' ? (r.name !== 'super_admin' && r.name !== 'admin') : true)
                      .map(r => (
                      <option key={r.name} value={r.name}>{r.name.replace('_', ' ')}</option>
                    ))}"""
code = code.replace(old_dropdown, new_dropdown)


# 4. Disable 'create_super_admin' checkbox for admin
old_checkbox = """                        <input
                          type="checkbox"
                          checked={permissions.includes(option.id)}
                          onChange={(e) => {"""

new_checkbox = """                        <input
                          type="checkbox"
                          disabled={currentUserRole === 'admin' && option.id === 'create_super_admin'}
                          checked={permissions.includes(option.id)}
                          onChange={(e) => {"""
code = code.replace(old_checkbox, new_checkbox)


# 5. Fix "Select All" button for admin so it doesn't select 'create_super_admin'
old_select_all = """                      onClick={() => {
                        if (permissions.length === SIDEBAR_OPTIONS.length) {
                          setPermissions([]);
                        } else {
                          setPermissions(SIDEBAR_OPTIONS.map(o => o.id));
                        }
                      }}"""

new_select_all = """                      onClick={() => {
                        const selectableOptions = currentUserRole === 'admin' 
                          ? SIDEBAR_OPTIONS.filter(o => o.id !== 'create_super_admin') 
                          : SIDEBAR_OPTIONS;
                        if (permissions.length >= selectableOptions.length) {
                          setPermissions([]);
                        } else {
                          setPermissions(selectableOptions.map(o => o.id));
                        }
                      }}"""
code = code.replace(old_select_all, new_select_all)


with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("done")
